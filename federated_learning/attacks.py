import random


def label_poisoning_attacks(num_labels, fed_args, dataset, model):
    """
    Perform label poisoning attacks by flipping the labels in the dataset.
    """
    if fed_args.attack == "static_label_flipping":
        def static_label_flip(examples):
            assert isinstance(examples["labels"], list)
            examples["labels"] = [
                num_labels - 1 - label for label in examples["labels"]
            ]
            return examples
        poisoned_dataset = dataset.map(static_label_flip, batched=True, load_from_cache_file=False, keep_in_memory=True)
    elif fed_args.attack == "random_label_flipping":
        def random_flip(examples):
            assert isinstance(examples["labels"], list)
            examples["labels"] = [
                random.randint(0, num_labels - 1)
                for _ in examples["labels"]
            ]
            return examples
        poisoned_dataset = dataset.map(random_flip, batched=True, load_from_cache_file=False, keep_in_memory=True)
    elif fed_args.attack == "dynamic_label_flipping":
        # Flip each label to the least-probable label according to the current model.
        # Use batched dataset.map (num_proc=1 so model closure isn't pickled).
        model.eval()
        device = next(model.parameters()).device

        from torch.nn.utils.rnn import pad_sequence
        import torch
        import torch.nn.functional as F
        
        pad_id = getattr(model.config, "pad_token_id", 0)
        def flip_batch(examples):
            assert isinstance(examples["labels"], list)

            # ---- pad input_ids ----
            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in examples["input_ids"]]
            input_ids = pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=pad_id
            ).to(device)
            
            # ---- pad attention_mask ----
            attention_mask = None
            if "attention_mask" in examples:
                mask_list = [torch.tensor(m, dtype=torch.long) for m in examples["attention_mask"]]
                attention_mask = pad_sequence(
                    mask_list,
                    batch_first=True,
                    padding_value=0
                ).to(device)

            # ---- forward to model ----
            with torch.no_grad():
                if attention_mask is not None:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids)
                logits = outputs.logits
                flipped = torch.argmin(logits, dim=-1).cpu().tolist()

            return {"labels": flipped}

        batch_size = min(32, len(dataset))
        poisoned_dataset = dataset.map(flip_batch, batched=True, batch_size=batch_size, load_from_cache_file=False, keep_in_memory=True) # WARNING: model-aware attack, must use num_proc=1 and GPU-safe
    else:
        raise ValueError(f"Unknown attack type: {fed_args.attack}")
    return poisoned_dataset


def label_poisoning_attacks_in_text_generation(fed_args, dataset, model):
    # poisoning all text tokens in response
    if fed_args.attack == "static_label_flipping":
        def static_label_flip(examples):
            assert isinstance(examples["response"], list)
            # flip the response using vocab size
            # Iterate over each response in the batch
            for i in range(len(examples["response"])):
                response = examples["response"][i]
                if isinstance(response, list) and len(response) > 0:
                    # Flip the last token
                    response = model.config.vocab_size - 1 - response
                    examples["response"][i] = response
            return examples
        poisoned_dataset = dataset.map(static_label_flip, batched=True, load_from_cache_file=False, keep_in_memory=True)
    elif fed_args.attack == "random_label_flipping":
        def random_label_flip(examples):
            assert isinstance(examples["response"], list)
            # flip the response using vocab size
            # Iterate over each response in the batch
            for i in range(len(examples["response"])):
                response = examples["response"][i]
                if isinstance(response, list) and len(response) > 0:
                    # Flip all the token of response randomly
                    response = [random.randint(0, model.config.vocab_size - 1) for _ in response]
                    examples["response"][i] = response
            return examples
        poisoned_dataset = dataset.map(random_label_flip, batched=True, load_from_cache_file=False, keep_in_memory=True)
    else:
        raise ValueError(f"Unknown attack type: {fed_args.attack}")
    return poisoned_dataset