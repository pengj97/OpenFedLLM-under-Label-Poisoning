import matplotlib.pyplot as plt
import numpy as np

root='/home/lingqing/pengj/OpenFedLLM/output'
model='Qwen2.5-Math-7B_lora'
dataset='state_tactic_pairs'
num_samples=56407
network='fed_n=8_b=1'
l=8192

# model='Llama-3.1-8B_lora'
# dataset='ag_news'
# # num_samples=7680
# num_samples=127600
# network='fed_n=4_b=1'
# l=8192

r=32
a=64
# split_strategy='non-iid'
# split_strategy='iid'
# split_strategy='dirichlet_alpha=1'
eval_interval=10

# lr=5e-5
lr=1e-4

# colors = ['green', 'red','orange', 'blue', 'purple', 'olive', 'grey', 'gold', ]
# markers = ['s', '+', 'v', '^',  'x',  'o', 'D', 'h', ]

colors = ['green',   'orange', 'blue', 'purple', 'olive', 'red']
markers = ['s',  'v',  '^', 'x', 'o', '+']

# colors = ['green',   'orange', 'blue', 'purple',  'red']
# markers = ['s',  'v',  '^', 'x',  '+']

# colors = [  'orange', 'blue', 'purple',  'red']
# markers = [ 'v',  '^', 'x',  '+']

# colors = [ 'red',  'orange',  'purple', 'olive']
# markers = ['+', 'v',  'x', 'o']

fed_algs = [
    ('fedavg', 'Baseline'),
    # ('fedavg', 'Baseline (FedAvg)'),
    # ('mean', 'Baseline (Mean)'),
    ('trimean', 'TriMean'),
    ('faba', 'FABA'),
    # 'cc_tau=0.5': 'CC',
    # ('cc_tau=0.1', 'CC'),
    ('cc_tau=0.01', 'CC'),
    ('lfighter', 'LFighter'),
    # ('faba-w', 'FABA-W'),
    # ('cc-w_tau=0.01','CC-W'),
    # ('lfighter-w', 'LFighter-W'),
    ('fedavg', 'Mean'),
    # ('fedavg', 'FedAvg'),
    # ('mean', 'Mean'),
]

split_strategies = [
    ('iid', 'IID'),
    ('dirichlet_alpha=1', 'Mild Noniid'),
    ('non-iid', 'Noniid'),
]
    
# fed_algs = [
#     ('fedavg', 'FedAvg'),
#     ('mean', 'Mean')
# ]


max_steps=1
batch_size=2
gradient_accumulation_steps=1
# attack='static_label_flipping'
# attack='random_label_flipping'
attack='dynamic_label_flipping'
# attack='None'

FONTSIZE=50

# pic_name = model + '_' + dataset + '_'  + network + '_'  + attack + '_' + split_strategy + f'_l{l}_b{batch_size}_lr{lr}'
pic_name = model + '_' + dataset + '_'  + network + '_'  + attack + f'_l{l}_b{batch_size}_lr{lr}'


fig, axes = plt.subplots(1, len(split_strategies), figsize=(24, 10), sharex=True, sharey=True)
axes[0].set_ylabel('Loss', fontsize=FONTSIZE)
# axes[0][0].set_ylim(0.3, 0.85)
# axes[0][0].set_ylim(0.3, 0.99)


for i in range(len(split_strategies)):
    # axes[i].set_title(split_strategies[i][1] + f' ({dataset.upper()})', fontsize=FONTSIZE)
    axes[i].set_title(split_strategies[i][1] + f' (STATE)', fontsize=FONTSIZE)
    axes[i].set_xlabel('Iterations', fontsize=FONTSIZE)
    axes[i].tick_params(labelsize=FONTSIZE)
    axes[i].grid('on')
    for agg_index, (fed_alg, alg_name) in enumerate(fed_algs):
        color = colors[agg_index]
        marker = markers[agg_index]
        if 'Baseline' in alg_name:
            # replace 'b=n' (n is arbitrary number) with 'b=0' in network
            network_baseline = network.split('_b=')[0] + '_b=0'
            file_path = f'{root}/{model}_r{r}a{a}_l{l}_{dataset}_{num_samples}/{network_baseline}/{split_strategies[0][0]}/{fed_alg}_i{max_steps}_b{batch_size}a{gradient_accumulation_steps}_None_lr{lr}'
        else:
            file_path = f'{root}/{model}_r{r}a{a}_l{l}_{dataset}_{num_samples}/{network}/{split_strategies[i][0]}/{fed_alg}_i{max_steps}_b{batch_size}a{gradient_accumulation_steps}_{attack}_lr{lr}'
        eval_loss_list = np.load(f'{file_path}/eval_loss_list.npy')

        rounds = list(range(0, (len(eval_loss_list)) * eval_interval, eval_interval))

        axes[i].plot(rounds, eval_loss_list, '-', label=alg_name, color=color, marker=marker, markevery=10, linewidth=4, markersize=10)
        # axes[0].legend()

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(fed_algs), fontsize=FONTSIZE-20, markerscale=2)

plt.subplots_adjust(top=0.91, bottom=0.28, left=0.09, right=0.95, hspace=0.23, wspace=0.22)

# plt.subplots_adjust(top=0.91, bottom=0.33, left=0.125, right=0.95, hspace=0.27, wspace=0.2)
# plt.tight_layout()

save_path_pdf = f'pic/pdf/{pic_name}.pdf'
save_path_png = f'pic/png/{pic_name}.png'
plt.savefig(save_path_pdf)
plt.savefig(save_path_png)