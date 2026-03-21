import random
import torch
import sklearn.metrics.pairwise as smp
from sklearn.cluster import KMeans
import numpy as np

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round


def flatten_dict_list(local_dict_list, device=None):
    if device is None:
        # default to CPU if no device specified
        device = torch.device('cpu')
    wList = [torch.cat([p.flatten().to(device) for p in local_dict.values()]) for local_dict in local_dict_list]
    wlist = torch.stack(wList)
    return wlist


def unflatten_to_dict(flat_tensor, reference_dict):
    new_dict = {}
    pointer = 0
    for key, param in reference_dict.items():
        param_size = param.numel()
        new_dict[key] = flat_tensor[pointer: pointer + param_size].view_as(param)
        pointer += param_size
    return new_dict
    

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
    elif fed_args.fed_alg == 'fedavg':   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    elif fed_args.fed_alg == 'trimean':
        poisoned_size = fed_args.num_poisoned_clients 
        for key in global_dict.keys():
            gradients_list = torch.stack([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round], dim=0)
            gradients_list = gradients_list.reshape(gradients_list.size(0), -1)
            sorted_gradients_list, _ = torch.sort(gradients_list, dim=0)
            
            # trim the largest and smallest 'poisoned_size' elements
            if poisoned_size == 0:
                trimmed_data = sorted_gradients_list
            elif poisoned_size > 0:
                trimmed_data = sorted_gradients_list[poisoned_size: -poisoned_size, :] 
            else:
                assert False, "Poisoned size should be equal or larger than 0!"
            
            # compute the average of trimmed_data
            if trimmed_data.nelement() > 0:
                tm = torch.mean(trimmed_data, dim=0)
            else:
                tm = 0
            tm = tm.view_as(global_dict[key])
            
            # gradient descent
            global_dict[key] = global_dict[key] + tm
    
            
    elif fed_args.fed_alg == 'faba':
        poisoned_size = fed_args.num_poisoned_clients
        # compute the local gradients of clients
        for key in global_dict.keys():
            for client in clients_this_round:
                local_dict_list[client][key] -= global_dict[key]

        # start faba aggregation
        # ensure concatenation happens on the same device as global parameters
        try:
            target_device = next(iter(global_dict.values())).device
        except StopIteration:
            target_device = torch.device('cpu')
        remain = flatten_dict_list(local_dict_list, device=target_device)
        for _ in range(poisoned_size):
            mean = remain.mean(dim=0)
            # remove the largest 'poisoned_size' model
            distances = torch.tensor([torch.norm(w - mean) for w in remain])
            remove_index = distances.argmax()
            remain = remain[torch.arange(len(remain)) != remove_index]
        
        faba_aggregate = remain.mean(dim=0)
        faba_aggregate = unflatten_to_dict(faba_aggregate, global_dict)

        # gradient descent: move each aggregate tensor to the parameter's device before adding
        for key in global_dict.keys():
            agg = faba_aggregate[key].to(global_dict[key].device)
            global_dict[key] = global_dict[key] + agg
    
    elif (fed_args.fed_alg).startswith('cc'):
        threshold = float(fed_args.fed_alg.split('=')[-1])
        # compute the local gradients of clients
        for key in global_dict.keys():
            for client in clients_this_round:
                local_dict_list[client][key] -= global_dict[key]

        # start cc aggregation
        # ensure concatenation happens on the same device as global parameters
        try:
            target_device = next(iter(global_dict.values())).device
        except StopIteration:
            target_device = torch.device('cpu')
        gradients_list = flatten_dict_list(local_dict_list, device=target_device)

        # start cc aggregation
        starting_point = torch.mean(gradients_list, dim=0)
        diff = torch.zeros_like(starting_point)
        for i in range(gradients_list.size(0)):
            norm  = torch.norm(gradients_list[i] - starting_point)
            if norm > threshold:
                diff += threshold / norm * (gradients_list[i] - starting_point)
            else:
                diff += (gradients_list[i] - starting_point)
        diff /=  gradients_list.size(0)
        cc_aggregate = starting_point + diff
        cc_aggregate = unflatten_to_dict(cc_aggregate, global_dict)

        # gradient descent: move each aggregate tensor to the parameter's device before adding
        for key in global_dict.keys():
            agg = cc_aggregate[key].to(global_dict[key].device)
            global_dict[key] = global_dict[key] + agg

    elif fed_args.fed_alg == 'lfighter':
        def clusters_dissimilarity(clusters):
            n0 = len(clusters[0])
            n1 = len(clusters[1])
            m = n0 + n1 
            cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
            cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
            mincs0 = np.min(cs0, axis=1)
            mincs1 = np.min(cs1, axis=1)
            ds0 = n0/m * (1 - np.mean(mincs0))
            ds1 = n1/m * (1 - np.mean(mincs1))
            return ds0, ds1

        # compute the local gradients of clients
        for key in global_dict.keys():
            for client in clients_this_round:
                local_dict_list[client][key] -= global_dict[key]

        # start lfighter aggregation
        num_clients = len(clients_this_round)
        dw = [[] for _ in range(num_clients)]
        for i in range(num_clients):
            # append the data of the last layer only
            last_key = list(global_dict.keys())[-2]
            dw[i].append(local_dict_list[i][last_key].cpu().data.numpy())
        dw = np.array(dw).squeeze()
        norms = np.linalg.norm(dw, axis=-1)
        memory = np.sum(norms, axis=0)
        # Compute potential source and target classes
        max_two_freq_classes = memory.argsort()[-2:]

        # gradient clustering
        data = []
        for i in range(num_clients):
            data.append(dw[i][max_two_freq_classes].reshape(-1))   

        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(data)
        labels = kmeans.labels_

        clusters = {0:[], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(data[i])

        # determine the "good" cluster if it has higher dissimilarity in clusters
        good_cl = 0
        cs0, cs1 = clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1
        good_clients = [i for i in range(num_clients) if labels[i] == good_cl]

        for key in global_dict.keys():
            global_dict[key] = global_dict[key] + \
             sum([local_dict_list[client][key] for client in good_clients]) / len(good_clients)
    else:
        raise NotImplementedError(f"Federated aggregation method {fed_args.fed_alg} is not implemented.")
    return global_dict, global_auxiliary