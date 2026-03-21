import torch
import copy 
import numpy as np

def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau**2
    elif fed_args.fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_auxiliary_dict(fed_args, global_dict):

    if fed_args.fed_alg in ['scaffold']:
        global_auxiliary = {}               # c in SCAFFOLD
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])
        auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # c_i in SCAFFOLD
        auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # delta c_i in SCAFFOLD

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*fed_args.num_clients
        auxiliary_delta_dict = [None]*fed_args.num_clients

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict


def average_state_dicts(local_state_dicts):
    """Element-wise average of PEFT (LoRA) state dicts."""
    n = len(local_state_dicts)
    if n == 0:
        raise ValueError("local_state_dicts is empty")

    keys = list(local_state_dicts[0].keys())
    for d in local_state_dicts[1:]:
        if set(d.keys()) != set(keys):
            raise ValueError("All local state dicts must share the same keys")

    avg_dict = {}
    with torch.no_grad():
        for k in keys:
            acc = torch.zeros_like(local_state_dicts[0][k])
            for i in range(n):
                acc.add_(local_state_dicts[i][k])
            acc.div_(float(n))
            avg_dict[k] = acc
    return avg_dict

def compute_metrics(p):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = p.label_ids
        pred_labels = np.argmax(preds, axis=1)
        accuracy = (pred_labels == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}