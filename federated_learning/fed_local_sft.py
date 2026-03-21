import torch
import copy
from trl import SFTTrainer
from transformers import TrainerCallback, Trainer, DataCollatorWithPadding
import torch
import numpy as np
from peft import get_peft_model_state_dict, set_peft_model_state_dict

def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    # elif (fed_args.fed_alg in ['fedavg', 'fedavgm', 'fedadgrad', 'fedyogi', 'fedadam']) or (fed_args.fed_alg).startswith('local'):
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    # else:
    #     raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer


def get_fed_local_classification_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, data_collator, global_state=None, local_auxiliary=None, global_auxiliary=None):
    """
    Return a Trainer configured for classification. Supports FedProx and SCAFFOLD by subclassing Trainer.
    """

    class TrainerFedProx(Trainer):
        def __init__(self, mu=0.0, global_state=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mu = mu
            self.global_state = global_state

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # FedProx regularization
            if self.mu and self.global_state is not None:
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    name_key = name.replace('.default', '')
                    if name_key in self.global_state:
                        loss = loss + self.mu / 2 * torch.norm(param - self.global_state[name_key]) ** 2

            return (loss, outputs) if return_outputs else loss

    class TrainerSCAFFOLD(TrainerFedProx):
        # For simplicity reuse FedProx loss and omit scaffold corrections here
        pass

    # choose trainer class based on fed_args
    def compute_metrics(p):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = p.label_ids
        pred_labels = np.argmax(preds, axis=1)
        accuracy = (pred_labels == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

    if fed_args.fed_alg == 'fedprox':
        trainer = TrainerFedProx(
            model=model,
            args=training_args,
            train_dataset=local_dataset,
            eval_dataset=local_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            mu=fed_args.prox_mu,
            global_state=global_state,
            compute_metrics=compute_metrics,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = TrainerSCAFFOLD(
            model=model,
            args=training_args,
            train_dataset=local_dataset,
            eval_dataset=local_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            global_state=global_state,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=local_dataset,
            eval_dataset=local_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)