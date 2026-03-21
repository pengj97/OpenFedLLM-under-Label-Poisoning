import torch
from .graph import MH_rule


def geometric_median(wList, max_iter=80, err=1e-5):
    # wList: (N, D)
    guess = wList.mean(dim=0)

    for _ in range(max_iter):
        # Compute Euclidean distances from current guess to each vector
        dist_li = torch.norm(wList - guess, dim=1)

        # Avoid division by zero without Python loops
        dist_li = torch.clamp(dist_li, min=1e-12)

        # Compute weights 1 / dist_li
        weights = 1.0 / dist_li       # shape (N,)

        # Compute weighted sum without stack()  
        # Broadcasting: (N, D) * (N, 1)
        temp1 = (wList * weights.unsqueeze(1)).sum(dim=0)

        temp2 = weights.sum()

        guess_next = temp1 / temp2

        # Check convergence
        if torch.norm(guess - guess_next) <= err:
            return guess_next

        guess = guess_next

    return guess


def smoothed_weiszfeld(
        wList, 
        alpha=None,        # shape (m,), default = 1
        nu=1e-5,           # smoothing parameter ν > 0
        max_iter=80, 
        tol=1e-5):
    """
    Smoothed Weiszfeld Algorithm for geometric median.

    Args:
        wList:  (m, d) tensor, each w_i is on a device in the paper
        alpha:  optional weights α_i, shape (m,)
        nu:     smoothing parameter (ν > 0)
        max_iter: number of iterations R
        tol:    convergence threshold

    Returns:
        v: geometric median approximation v^(R)
    """

    m, d = wList.shape

    # if alpha_i not provided, set them all to 1
    if alpha is None:
        alpha = torch.ones(m, device=wList.device, dtype=wList.dtype)

    # initial v^(0)
    v = wList.mean(dim=0)

    for _ in range(max_iter):

        # Compute ||v - w_i||
        dist = torch.norm(wList - v, dim=1)  # shape (m,)

        # β_i^(r) = α_i / max(ν, dist)
        denom = torch.maximum(dist, torch.tensor(nu, device=wList.device, dtype=wList.dtype))
        beta = alpha / denom  # shape (m,)

        # v^(r+1) = (sum_i beta_i w_i) / (sum_i beta_i)
        numerator = (wList * beta.unsqueeze(1)).sum(dim=0)
        denominator = beta.sum()
        v_next = numerator / denominator

        # convergence check
        if torch.norm(v - v_next) < tol:
            return v_next

        v = v_next

    return v



def Krum_index(wList, byzantine_size):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=torch.float, device=wList.device)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()**2
            # We need minimized distance so we add a minus sign here
            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    # The distance from any node to itself must be 0.00, so we add 1 here
    k = node_size - byzantine_size - 2 + 1
    topv, _ = dist.topk(k=k, dim=1)
    scores = topv.sum(dim=1)
    return scores.argmax()


def Krum(wList, byzantine_size):
    index = Krum_index(wList, byzantine_size)
    return wList[index]


# def trimmed_mean(wList, byzantine_size):
#     node_size = wList.size(0)
#     if node_size == 2 * byzantine_size:
#         return 0
#     proportion_to_cut = byzantine_size / node_size
#     tm_np = stats.trim_mean(wList.cpu(), proportion_to_cut, axis=0)
#     return torch.from_numpy(tm_np).to(DEVICE)


def trimmed_mean(wList, byzantine_size):
    # 将张量按某个维度排序
    sorted_wList, _ = torch.sort(wList, dim=0)
    
    # 对排序后的张量进行修剪
    if byzantine_size == 0:
        trimmed_data = sorted_wList
    elif byzantine_size > 0:
        trimmed_data = sorted_wList[byzantine_size:-byzantine_size, :]
    else:
        assert False, 'Poisoned size should be equal or larger than 0!'
    
    # 计算修剪后的均值
    if trimmed_data.nelement() > 0:
        tm = torch.mean(trimmed_data, dim=0)
    else:
        tm = 0

    return tm


def faba(wList, byzantine_size):
    remain = wList
    for _ in range(byzantine_size):
        mean = remain.mean(dim=0)
        # remove the largest 'byzantine_size' model
        distances = torch.tensor([
            torch.norm(model - mean) for model in remain
        ])
        remove_index = distances.argmax()
        remain = remain[torch.arange(remain.size(0)) != remove_index]
    if remain.size(0) == 0:
        return 0
    else:
        return remain.mean(dim=0)

def ios(wList, weightList, byzantine_size):
    remain_models = wList
    remain_weights = weightList
    for _ in range(byzantine_size):
        mean = torch.tensordot(remain_weights, remain_models, dims=1)
        # remove the largest 'byzantine_size' model
        distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
        remove_idx = distances.argmax()
        remain_idx = torch.arange(remain_models.size(0)) != remove_idx
        remain_models = remain_models[remain_idx]
        remain_weights = remain_weights[remain_idx]
    res = torch.tensordot(remain_weights, remain_models, dims=1)
    res /= remain_weights.sum()
    return res

def cc(wList, threshold):
    starting_point = torch.mean(wList, dim=0)
    diff = torch.zeros_like(starting_point)
    for i in range(wList.size(0)):
        norm  = torch.norm(wList[i] - starting_point)
        if norm > threshold:
            diff += threshold / norm * (wList[i] - starting_point)
        else:
            diff += (wList[i] - starting_point)
    diff /=  wList.size(0)
    cc_aggregate = starting_point + diff
    return cc_aggregate

def cg(wList, weight_list, threshold):
    local_model = wList[-1]
    cum_diff = torch.zeros_like(local_model)
    for n in range(len(wList) - 1):
        model = wList[n]
        diff = model - local_model
        norm = diff.norm()
        weight = weight_list[n]
        if norm > threshold:
                cum_diff += weight * threshold * diff / norm
        else:
                cum_diff += weight * diff
    return local_model + cum_diff


# Decentralized aggregation.
def dec_aggregate(fed_args, client_id, clients_this_round, local_dict_list, sample_num_list, G):
    n_clients = len(local_dict_list)
    # build mixing matrix
    W = MH_rule(G)
    # ensure device matches parameters
    if isinstance(W, torch.Tensor) and n_clients > 0 and len(local_dict_list) > 0:
        try:
            device = next(iter(local_dict_list[0].values())).device
        except StopIteration:
            device = torch.device('cpu')
        W = W.to(device)

    # prepare new local dict (do not modify input dicts in-place)
    reference_dict = local_dict_list[client_id]
    new_local_dict = {}

    # default: if no clients sampled, return copy of local
    if len(clients_this_round) == 0:
        for k, v in reference_dict.items():
            new_local_dict[k] = v.clone()
        return new_local_dict

    if fed_args.fed_alg == 'meanW':
        for key, ref_param in reference_dict.items():
            # initialize accumumulator on the correct device and dtype
            accum = torch.zeros_like(ref_param, device=ref_param.device)
            for j in clients_this_round:
                param = local_dict_list[j][key]
                # mixing weight from row client_id, column j
                try:
                    w = float(W[client_id, j]) if isinstance(W, torch.Tensor) else float(W[client_id][j])
                except Exception:
                    w = 0.0
                accum += param * w
            new_local_dict[key] = accum
    elif fed_args.fed_alg == 'mean':
        for key, ref_param in reference_dict.items():
            accum = torch.zeros_like(ref_param, device=ref_param.device)
            for j in clients_this_round:
                param = local_dict_list[j][key]
                accum += param
            new_local_dict[key] = accum / len(clients_this_round)
    elif fed_args.fed_alg == 'trimean':
        for key, ref_param in reference_dict.items():
            # include neighbors plus self parameter tensor
            neighbor_list = [local_dict_list[j][key] for j in G.neighbors[client_id]]
            neighbor_list.append(ref_param)
            neighbor_params = torch.stack(neighbor_list, dim=0)
            neighbor_params = neighbor_params.reshape(neighbor_params.size(0), -1)
            tm = trimmed_mean(neighbor_params, G.byzantine_sizes[client_id])
            new_local_dict[key] = tm.view_as(ref_param)
    elif fed_args.fed_alg == 'rfa':
        for key, ref_param in reference_dict.items():
            # include neighbors plus self parameter tensor
            neighbor_list = [local_dict_list[j][key] for j in G.neighbors[client_id]]
            neighbor_list.append(ref_param)
            neighbor_params = torch.stack(neighbor_list, dim=0)
            neighbor_params = neighbor_params.reshape(neighbor_params.size(0), -1)
            gm = smoothed_weiszfeld(neighbor_params, nu=1e-5, max_iter=80, tol=1e-5)
            new_local_dict[key] = gm.view_as(ref_param)
    elif fed_args.fed_alg == 'faba':
        for key, ref_param in reference_dict.items():
            neighbor_list = [local_dict_list[j][key] for j in G.neighbors[client_id]]
            neighbor_list.append(ref_param)
            neighbor_params = torch.stack(neighbor_list, dim=0)
            neighbor_params = neighbor_params.reshape(neighbor_params.size(0), -1)
            faba_agg = faba(neighbor_params, G.byzantine_sizes[client_id])
            new_local_dict[key] = faba_agg.view_as(ref_param)
    elif (fed_args.fed_alg).startswith('cc'):
        threshold = float(fed_args.fed_alg.split('=')[-1])
        for key, ref_param in reference_dict.items():
            neighbor_list = [local_dict_list[j][key] for j in G.neighbors[client_id]]
            neighbor_list.append(ref_param)
            neighbor_params = torch.stack(neighbor_list, dim=0)
            neighbor_params = neighbor_params.reshape(neighbor_params.size(0), -1)
            cc_agg = cc(neighbor_params, threshold)
            new_local_dict[key] = cc_agg.view_as(ref_param)
    elif fed_args.fed_alg == 'ios':
        neighbors = G.neighbors[client_id]
        # fetch weights for current client and its neighbors; keep as tensor when possible
        if isinstance(W, torch.Tensor):
            weight_list = W[client_id, neighbors]
            # append self-weight as a new element
            self_w = W[client_id, client_id]
            weight_list = torch.cat([weight_list, self_w.unsqueeze(0)])
        else:
            # assume W is a nested list or similar
            weight_list = list(W[client_id][neighbors])
            weight_list.append(W[client_id][client_id])
        for key, ref_param in reference_dict.items():
            neighbor_list = [local_dict_list[j][key] for j in G.neighbors[client_id]]
            neighbor_list.append(ref_param)
            neighbor_params = torch.stack(neighbor_list, dim=0)
            neighbor_params = neighbor_params.reshape(neighbor_params.size(0), -1)
            ios_agg = ios(neighbor_params, weight_list, G.byzantine_sizes[client_id])
            new_local_dict[key] = ios_agg.view_as(ref_param)
    elif (fed_args.fed_alg).startswith('cg'):
        threshold = float(fed_args.fed_alg.split('=')[-1])
        neighbors = G.neighbors[client_id]
        if isinstance(W, torch.Tensor):
            weight_list = W[client_id, neighbors]
            self_w = W[client_id, client_id]
            weight_list = torch.cat([weight_list, self_w.unsqueeze(0)])
        else:
            weight_list = list(W[client_id][neighbors])
            weight_list.append(W[client_id][client_id])
        for key, ref_param in reference_dict.items():
            neighbor_list = [local_dict_list[j][key] for j in G.neighbors[client_id]]
            neighbor_list.append(ref_param)
            neighbor_params = torch.stack(neighbor_list, dim=0)
            neighbor_params = neighbor_params.reshape(neighbor_params.size(0), -1)
            cg_agg = cg(neighbor_params, weight_list, threshold)
            new_local_dict[key] = cg_agg.view_as(ref_param)
    else:
        raise ValueError(f"Unsupported aggregation algorithm: {fed_args.fed_alg}")
    return new_local_dict

