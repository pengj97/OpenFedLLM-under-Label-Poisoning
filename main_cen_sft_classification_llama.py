import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding


from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
# Set global seeds for reproducibility
set_all_seeds(script_args.seed)
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
# dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# Import tokenizer early to map columns and set pad_token if needed
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    # Prefer setting pad_token to eos_token if present, otherwise fall back to unk_token
    if getattr(tokenizer, 'eos_token', None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif getattr(tokenizer, 'unk_token', None) is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # As a last resort, add a pad token (will increase vocab size)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"Tokenizer pad_token set to: {tokenizer.pad_token}")

# Tokenize dataseet for classification
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=script_args.seq_length)

# Tokenize 数据集
# dataset = dataset.map(tokenize_fn, batched=True)

# dataset = dataset.rename_column("label", "labels")
# train_dataset = dataset["train"]
# eval_dataset = dataset["test"]
# num_labels= len(set(train_dataset['labels']))

# Tokenize 数据集
dataset = dataset.map(tokenize_fn, batched=True)

dataset = dataset.rename_column("label", "labels")
train_dataset = dataset["train"]
train_dataset = train_dataset.shuffle(seed=script_args.seed) 
num_labels= len(set(train_dataset['labels']))

few_shot_per_label = 20
counts = [0] *  num_labels
selected_indices = []
for i in range(len(train_dataset)):
    if counts[train_dataset[i]['labels']] < few_shot_per_label:
        counts[train_dataset[i]['labels']] += 1
        selected_indices.append(i)
train_dataset = train_dataset.select(selected_indices)
eval_dataset = dataset["test"]

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, train_dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    # device_map=device_map,
    device_map="auto",
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    num_labels=num_labels,
)

# Ensure model config knows the pad token id (some model forwards require this)
if getattr(tokenizer, "pad_token_id", None) is not None:
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token = tokenizer.pad_token
    except Exception:
        pass

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

# If using PEFT, ensure task type is sequence classification
from peft import TaskType
if peft_config is not None:
    try:
        peft_config.task_type = TaskType.SEQ_CLS
    except Exception:
        pass

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))

local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the formatting function =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Start federated training =====
eval_accuracy_list = []
eval_loss_list = []
regular_clients = list(range(fed_args.num_clients - fed_args.num_poisoned_clients))
poisoned_clients = list(range(fed_args.num_clients - fed_args.num_poisoned_clients, fed_args.num_clients))


print('=========================================================')
print('[Task] Sequence Classification Federated Learning')
print('=========================================================')
print('[Setting]')
print('{:12s} model={} seq_length={}'.format('[task]', script_args.model_name_or_path.split('/')[-1], script_args.seq_length))
print('{:12s} dataset={} num_labels={} partition={}'.format(
    '[dataset]', script_args.dataset_name, num_labels, fed_args.split_strategy))
print('{:12s} name={} attack={} use_peft={}'.format(
    '[Algorithm]', fed_args.fed_alg, fed_args.attack, script_args.use_peft))
print('{:12s} lr={} lr_ctrl={}, batch_size={}'.format(
    '[Optimizer]', script_args.learning_rate, 'cosine_lr', script_args.batch_size))
print('{:12s} size={}, regular_size={}, poisoned_size={}'.format(
    '[Clients]', fed_args.num_clients, fed_args.num_clients - fed_args.num_poisoned_clients, fed_args.num_poisoned_clients))
print('{:12s} rounds={}, eval_interval={}'.format(
    '[Running]', fed_args.num_rounds, fed_args.eval_interval))
print('{:12s} seed={}, fix_seed={}'.format('[Randomness]', script_args.seed,'True'))
print('-------------------------------------------')

print('len of local datasets: {}'.format(
    [len(local_datasets[i]) for i in range(fed_args.num_clients)]
))
print('sets of labels of clients: {}'.format(
    [set(local_datasets[i]['labels']) for i in range(fed_args.num_clients)]
))

for round in tqdm(range(fed_args.num_rounds+1)):

    clients_this_round = get_clients_this_round(fed_args, round)
    regular_clients_this_round = [c for c in clients_this_round if c in regular_clients]
    poisoned_clients_this_round = [c for c in clients_this_round if c in poisoned_clients]

    new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
    training_args = get_training_args(script_args, new_lr)

    print(f">> ==================== Round {round} : Regular: {regular_clients_this_round} Poisoned: {poisoned_clients_this_round} lr: {new_lr}  ====================")
    
    for client in range(fed_args.num_clients):
        # sync the global model to the local model
        set_peft_model_state_dict(model, global_dict)
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        
        # conducting label poisoning attacks if specified
        if fed_args.attack is not None:
            if client in poisoned_clients_this_round:
                sub_dataset = label_poisoning_attacks(num_labels, fed_args, sub_dataset, model)

        # ===== Train local model on the client side (classification) =====
        trainer = get_fed_local_classification_trainer(
            script_args=script_args,
            fed_args=fed_args,
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        # Since clients' local models are synced with the global model,
        # only evaluate on one client to save time
        if client == 0:
            if round % fed_args.eval_interval == 0:
                eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
                eval_accuracy_list.append(eval_metrics['eval_accuracy'])
                eval_loss_list.append(eval_metrics['eval_loss'])
                print(eval_metrics, flush=True)

        trainer.train()

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    # Update global model
    if peft_config is not None:
        set_peft_model_state_dict(model, global_dict)
    else:
        model.load_state_dict(global_dict, strict=False)

    # ===== Save the model =====
    if round % fed_args.save_model_freq == 0 and round > 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round}"))


np.save(os.path.join(script_args.output_dir, "eval_accuracy_list.npy"), np.array(eval_accuracy_list))
np.save(os.path.join(script_args.output_dir, "eval_loss_list.npy"), np.array(eval_loss_list))
print("Evaluation accuracy list:", eval_accuracy_list)
print("Evaluation loss list:", eval_loss_list)
