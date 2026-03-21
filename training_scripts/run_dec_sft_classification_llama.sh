#!/bin/bash
max_steps=1
num_rounds=1000
batch_size=2
gradient_accumulation_steps=1
seq_length=8192
num_clients=4
sample_clients=4
num_poisoned_clients=1
lora_r=32
lora_alpha=64  # twice of lora_r
lr=1e-4
seed=2025
eval_interval=10

local_data_dir="./data/"
dataset_name="ag_news"
# dataset_sample=127600
dataset_sample=7680
model_name_or_path="/home/lingqing/pengj/Llama-3.1-8B"
output_dir=./output

gpu=1,2,3,4

fed_alg="meanW"
# fed_alg="trimean"
# fed_alg="faba"
# fed_alg="cc_tau=0.01"
# fed_alg="ios"
# fed_alg="cg_tau=0.01"
# fed_alg="rfa"

split_strategies=("iid" "dirichlet_alpha=1" "non-iid")
attacks=("static_label_flipping" "dynamic_label_flipping")

graph="Fan"


for attack in "${attacks[@]}"; do
    for split_strategy in "${split_strategies[@]}"; do
        echo "====== Running attack = $attack, split_strategy = $split_strategy ======"

        log_file=logs/sft/Llama-3.1-8B/"$graph"_n="$num_clients"_b="$num_poisoned_clients"/"$attack"/"$fed_alg"_"$attack"_"$split_strategy".log
        # make sure the log directory exists before running
        log_dir=$(dirname "$log_file")
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
        fi

        CUDA_VISIBLE_DEVICES=$gpu nohup python main_dec_sft_classification_llama.py \
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
            --attack $attack \
            --split_strategy $split_strategy \
            --graph $graph \
            --seed $seed \
            --eval_interval $eval_interval \
            --template "qwen" \
            > "$log_file" 2>&1 &

        pid=$!   # 获取后台进程 PID
        echo "Started PID: $pid (log: $log_file)"

        # 阻塞等待该进程结束，再进入下一个循环
        wait $pid
        echo "Completed: $attack, $split_strategy"
    done
done

echo "All tasks completed."