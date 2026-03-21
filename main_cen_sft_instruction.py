import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
# from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
dataset = dataset.rename_column("tactic", "labels")

split = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, train_dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    # device_map=device_map,
    device_map="auto",
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

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

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# Process eval_dataset: format and tokenize it
def preprocess_eval_dataset(example):
    # Create a batch with a single example for formatting_prompts_func
    batch_example = {
        'instruction': [example['instruction']],
        'response': [example['response']]
    }
    formatted_texts = formatting_prompts_func(batch_example)
    # formatted_texts is a list with one string
    tokens = tokenizer(
        formatted_texts[0],
        truncation=True,
        max_length=script_args.seq_length,
        padding='max_length',  # Pad to max_length
    )
    # For causal LM, labels should be same as input_ids
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

eval_dataset_processed = eval_dataset.map(
    preprocess_eval_dataset,
    batched=False,
    remove_columns=eval_dataset.column_names,
)

# ===== Start federated training =====
eval_loss_list = []
regular_clients = list(range(fed_args.num_clients - fed_args.num_poisoned_clients))
poisoned_clients = list(range(fed_args.num_clients - fed_args.num_poisoned_clients, fed_args.num_clients))


print('=========================================================')
print('[Task] Text Generation Federated Learning')
print('=========================================================')
print('[Setting]')
print('{:12s} model={} seq_length={}'.format('[task]', script_args.model_name_or_path.split('/')[-1], script_args.seq_length))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', script_args.dataset_name, fed_args.split_strategy))
print('{:12s} name={} attack={} use_peft=True'.format(
    '[Algorithm]', fed_args.fed_alg, fed_args.attack))
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

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        
        # conducting label poisoning attacks if specified
        if fed_args.attack is not None:
            if client in poisoned_clients_this_round:
                sub_dataset = label_poisoning_attacks_in_text_generation(fed_args, sub_dataset, model)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        # Compute the loss of the global model (the local model 0 is the same as the global model before training)
        # on the dataset before training
        if client == 0:
            if round % fed_args.eval_interval == 0:
                eval_metrics = trainer.evaluate(eval_dataset=eval_dataset_processed)
                eval_loss = eval_metrics['eval_loss']
                eval_loss_list.append(eval_loss)
                print(eval_metrics, flush=True)

        # ===== Train local model on the client side =====
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
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if round % fed_args.save_model_freq == 0 and round > 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round}"))

np.save(os.path.join(script_args.output_dir, "eval_loss_list.npy"), np.array(eval_loss_list))
print("Evaluation loss list:", eval_loss_list)
