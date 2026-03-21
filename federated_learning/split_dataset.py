import random
import numpy as np

def split_dataset(fed_args, script_args, dataset):
    local_datasets = []
    if fed_args.split_strategy == "iid":
        dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))

    elif fed_args.split_strategy == "non-iid":        
        labels = list(dataset['labels'])
        unique_labels = sorted(set(labels))
        num_clients = fed_args.num_clients
        num_classes = len(unique_labels)
        average_data_per_client = np.ceil(len(dataset) / num_clients).astype(int)

        label_to_indices = {lab: [] for lab in unique_labels}
        for idx, lab in enumerate(labels):
            label_to_indices[lab].append(idx)

        partition = [[] for _ in range(num_clients)]
        if num_classes >= num_clients:
            # class_cnt >= node_cnt, some nodes possess several classes
            for lab, indices in label_to_indices.items():
                client_id = lab % num_clients
                partition[client_id].extend(indices)
        else:
            # class_cnt < node_cnt, some classes are allocated on different workers
            group_boundary = [(group_idx*num_clients) // num_classes for group_idx in range(num_classes)]
            insert_node_ptrs = group_boundary.copy()
            group_boundary.append(num_clients)
            for data_idx, lab in enumerate(labels):
                group_idx = unique_labels.index(lab)
                client_id = insert_node_ptrs[group_idx]
                partition[client_id].append(data_idx)
                # `insert_node_ptrs[group_idx]` increases by 1
                if insert_node_ptrs[group_idx] + 1 < group_boundary[group_idx+1]:
                    insert_node_ptrs[group_idx] += 1
                else:
                    insert_node_ptrs[group_idx] = group_boundary[group_idx]
        for client_idx in range(num_clients):
            client_indices = partition[client_idx]
            local_datasets.append(dataset.select(client_indices))

    elif fed_args.split_strategy == "non-iid-equal-size":        
        labels = list(dataset['labels'])
        unique_labels = sorted(set(labels))
        num_clients = fed_args.num_clients
        num_classes = len(unique_labels)
        average_data_per_client = np.ceil(len(dataset) / num_clients).astype(int)

        label_to_indices = {lab: [] for lab in unique_labels}
        for idx, lab in enumerate(labels):
            label_to_indices[lab].append(idx)

        partition = [[] for _ in range(num_clients)]
        if num_classes >= num_clients:
            # class_cnt >= node_cnt, some nodes possess several classes
            for lab, indices in label_to_indices.items():
                client_id = lab % num_clients
                # if len(partition[client_id]) <= average_data_per_client, assign indices to partition[client_id], else find the next client that satisfies the condition
                for idx in indices:
                    while len(partition[client_id]) >= average_data_per_client:
                        client_id = (client_id + 1) % num_clients
                    partition[client_id].append(idx)

        else:
            # class_cnt < node_cnt, some classes are allocated on different workers
            group_boundary = [(group_idx*num_clients) // num_classes for group_idx in range(num_classes)]
            insert_node_ptrs = group_boundary.copy()
            group_boundary.append(num_clients)
            for data_idx, lab in enumerate(labels):
                group_idx = unique_labels.index(lab)
                client_id = insert_node_ptrs[group_idx]
                while len(partition[client_id]) >= average_data_per_client:
                    client_id = (client_id + 1) % num_clients
                partition[client_id].append(data_idx)
                # `insert_node_ptrs[group_idx]` increases by 1
                if insert_node_ptrs[group_idx] + 1 < group_boundary[group_idx+1]:
                    insert_node_ptrs[group_idx] += 1
                else:
                    insert_node_ptrs[group_idx] = group_boundary[group_idx]
        for client_idx in range(num_clients):
            client_indices = partition[client_idx]
            local_datasets.append(dataset.select(client_indices))

    elif fed_args.split_strategy.startswith("dirichlet"):
        alpha = float(fed_args.split_strategy.split("=")[-1])
        labels = list(dataset['labels'])
        unique_labels = sorted(set(labels))
        num_clients = fed_args.num_clients
        num_classes = len(unique_labels)
        label_to_indices = {lab: [] for lab in unique_labels}
        for idx, lab in enumerate(labels):
            label_to_indices[lab].append(idx)
        min_size = len(dataset) // (num_clients * 2)
        
        current_min_size = 0
        while current_min_size < min_size:
            partition = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = label_to_indices[k]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # using the proportions from dirichlet, only select those nodes having data amount less than average
                proportions = np.array(
                    [p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, partition)]
                )
                # scale proportions to sum to 1
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)
                partition = [idx_j + idx.tolist() for idx_j, idx in zip(partition, np.split(idx_k, proportions))]
                current_min_size = min([len(idx_j) for idx_j in partition])
        for client_idx in range(num_clients):
            client_indices = partition[client_idx]
            local_datasets.append(dataset.select(client_indices))
    else:
        raise ValueError(f"Unknown split strategy: {fed_args.attack}")

    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round