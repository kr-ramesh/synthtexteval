#!/bin/bash
#SBATCH --job-name="wiki-bio-gen"
#SBATCH --time=96:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:4
#SBATCH --partition=fastgpus
#SBATCH --output=text-outputs/continual-training-dp-4.txt

path_to_dataset="/data/datasets/wikipedia-biographies-v1->-200.csv"
data_dir="/data/dp-fact/text-gen/" 
path_to_model="/data/dp-fact/text-gen/models/princeton_wiki_updated_DP_4_30_epochs_" 
enable_dp=true
epsilon_value=4
epochs=15
gradient_accumulation_steps=64
load_from_ckpt=true
path_to_ckpt="/data/dp-fact/text-gen/models/princeton_wiki_updated_DP_4"

python -m torch.distributed.run --nproc_per_node 4 lora_dp_trainer.py \
        --output_dir outputs \
        --model_name princeton \
        --path_to_dataset $path_to_dataset \
        --path_to_save $path_to_model \
        --enable_dp $enable_dp \
        --target_epsilon $epsilon_value \
        --load_from_ckpt $load_from_ckpt \
        --path_to_load $path_to_ckpt \
        --target_delta 1e-5 \
        --dataset_name wiki \
        --sequence_len 1024 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --evaluation_strategy no \
        --save_strategy "epoch" \
        --log_level info \
        --seed 42 \
        --per_sample_max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --remove_unused_columns False \
        --num_train_epochs $epochs \
        --logging_steps 4 \
        --max_grad_norm 0 \
        --lr_scheduler_type constant \
        --learning_rate 3e-4 \
        --disable_tqdm True \
        --dataloader_num_workers 2 \
        --lora_dim 4 \
        --lora_alpha 32 \
        --lora_dropout 0.0 \
        --enable_lora \
        --target_modules "['q_proj', 'v_proj']" \
        --label_names labels \
        --gradient_checkpointing \
        --num_codes 1000 \
        --is_balanced
