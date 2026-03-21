import matplotlib.pyplot as plt
import numpy as np

root='/home/lingqing/pengj/OpenFedLLM/output'

model='Qwen2.5-Math-7B_lora'
l=8192
r=32
a=64
dataset='state_tactic_pairs'
num_samples=56407
# network='fed_n=6_b=1'
network='fed_n=8_b=1'
split_strategy='non-iid'
# split_strategy='iid'
# split_strategy='dirichlet_alpha=1'
eval_interval=10

# lr=5e-5
lr=1e-4

colors = ['green', 'red','orange', 'blue', 'purple', 'olive', 'grey', 'gold', ]
markers = ['s', '+', 'v', '^',  'x',  'o', 'D', 'h', ]

# colors = ['green', 'red',  'orange', 'blue', 'purple', 'olive']
# markers = ['s', '+', 'v',  '^', 'x', 'o']

# colors = [ 'red',  'orange', 'blue', 'purple', 'olive']
# markers = ['+', 'v',  '^', 'x', 'o']

# colors = [ 'red',   'blue', 'purple', 'olive']
# markers = ['+',   '^', 'x', 'o']

# colors = [ 'red',  'orange',  'purple', 'olive']
# markers = ['+', 'v',  'x', 'o']

fed_algs = [
    ('fedavg', 'Baseline (FedAvg)'),
    # ('mean', 'Baseline (Mean)'),
    ('fedavg', 'FedAvg'),
    # ('mean', 'Mean'),
    ('trimean', 'TriMean'),
    ('faba', 'FABA'),
    # 'cc_tau=0.5': 'CC',
    # ('cc_tau=0.1', 'CC'),
    ('cc_tau=0.01', 'CC'),
    ('lfighter', 'LFighter')
    # ('faba-w', 'FABA-W'),
    # ('cc-w_tau=0.01','CC-W'),
    # ('lfighter-w', 'LFighter-W'),
]
    
# fed_algs = [
#     ('fedavg', 'FedAvg'),
#     ('mean', 'Mean')
# ]


max_steps=1
batch_size=2
gradient_accumulation_steps=1
attack='static_label_flipping'
# attack='dynamic_label_flipping'
# attack='None'

FONTSIZE=50


pic_name = model + '_' + dataset + '_' + network + '_'  + attack + '_' + split_strategy + f'_l{l}_b{batch_size}_lr{lr}'

fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
axes[0].set_ylabel('Accuracy', fontsize=FONTSIZE)
axes[1].set_ylabel('Loss', fontsize=FONTSIZE)
axes[0].set_xlabel('Iterations', fontsize=FONTSIZE)
axes[1].set_xlabel('Iterations', fontsize=FONTSIZE)
axes[0].set_title('Evaluation Accuracy', fontsize=FONTSIZE)
axes[1].set_title('Evaluation Loss', fontsize=FONTSIZE)
axes[0].tick_params(labelsize=FONTSIZE)
axes[1].tick_params(labelsize=FONTSIZE)
axes[0].grid('on')
axes[1].grid('on')


for agg_index, (fed_alg, alg_name) in enumerate(fed_algs):
    color = colors[agg_index]
    marker = markers[agg_index]
    if 'Baseline' in alg_name:
        # replace 'b=n' (n is arbitrary number) with 'b=0' in network
        network_baseline = network.split('_b=')[0] + '_b=0'
        file_path = f'{root}/{model}_r{r}a{a}_l{l}_{dataset}_{num_samples}/{network_baseline}/{split_strategy}/{fed_alg}_i{max_steps}_b{batch_size}a{gradient_accumulation_steps}_None_lr{lr}'
    else:
        file_path = f'{root}/{model}_r{r}a{a}_l{l}_{dataset}_{num_samples}/{network}/{split_strategy}/{fed_alg}_i{max_steps}_b{batch_size}a{gradient_accumulation_steps}_{attack}_lr{lr}'
    eval_accuracy_list = np.load(f'{file_path}/eval_accuracy_list.npy')
    eval_loss_list = np.load(f'{file_path}/eval_loss_list.npy')

    rounds = list(range(0, (len(eval_accuracy_list)) * eval_interval, eval_interval))

    axes[0].plot(rounds, eval_accuracy_list, '-', label=alg_name, color=color, marker=marker, markevery=10, linewidth=4, markersize=10)
    axes[1].plot(rounds, eval_loss_list, '-', label=alg_name, color=color, marker=marker, markevery=10, linewidth=4, markersize=10)
    # axes[0].legend()

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(fed_algs)//2, fontsize=FONTSIZE-20, markerscale=2)

plt.subplots_adjust(top=0.91, bottom=0.33, left=0.125, right=0.95, hspace=0.27, wspace=0.2)
# plt.tight_layout()

save_path_pdf = f'pic/pdf/{pic_name}.pdf'
save_path_png = f'pic/png/{pic_name}.png'
plt.savefig(save_path_pdf)
plt.savefig(save_path_png)