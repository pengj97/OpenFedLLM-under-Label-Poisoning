#!/bin/bash
max_steps=1
num_rounds=500
batch_size=2
gradient_accumulation_steps=1
seq_length=2048
num_clients=8
sample_clients=8
num_poisoned_clients=0
lora_r=32
lora_alpha=64  # twice of lora_r
lr=1e-4
seed=2025
eval_interval=10

local_data_dir="./data/"
dataset_name="state_tactic_pairs"
dataset_sample=56407
model_name_or_path="/home/lingqing/pengj/Qwen2.5-Math-7B"
output_dir=./output

gpu=1,2,3,4
fed_alg="fedavg"
attack=None
split_strategies=("iid" "dirichlet_alpha=1" "non-iid")


for split_strategy in "${split_strategies[@]}"; do
    echo "====== Running split_strategy = $split_strategy ======"

    log_file=logs/sft_instruct/Qwen2.5-Math-7B/"cen"_n="$num_clients"_b="$num_poisoned_clients"/"$attack"/"$fed_alg"_"$attack"_"$split_strategy".log

    mkdir -p "$(dirname "$log_file")"

    CUDA_VISIBLE_DEVICES=$gpu nohup python main_cen_sft_instruction.py \
        --learning_rate $lr \
        --model_name_or_path $model_name_or_path \
        --local_data_dir $local_data_dir \
        --dataset_name $dataset_name \
        --dataset_sample $dataset_sample \
        --fed_alg $fed_alg \
        --num_clients $num_clients \
        --sample_clients $sample_clients \
        --num_poisoned_clients $num_poisoned_clients \
        --max_steps $max_steps \
        --num_rounds $num_rounds \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --seq_length $seq_length \
        --peft_lora_r $lora_r \
        --peft_lora_alpha $lora_alpha \
        --use_peft \
        --trust_remote_code True \
        --output_dir $output_dir \
        --split_strategy $split_strategy \
        --seed $seed \
        --eval_interval $eval_interval \
        --template "qwen" \
        > "$log_file" 2>&1 &

    pid=$!   # 获取后台进程 PID
    echo "Started PID: $pid (log: $log_file)"
    
    # 阻塞等待该进程结束，再进入下一个循环
    wait $pid
    echo "Completed: $split_strategy"
done

echo "All tasks completed."