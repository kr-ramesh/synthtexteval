export data_dir=$1
export path_to_model=$2
export enable_dp=$3
export epsilon_value=$4
export path_to_dataset=$5
export epochs=$6
export gradient_accumulation_steps=$7
#export load_ckpt=$8
#export path_to_ckpt=$9


data_dir=${data_dir:-"/data/dp-fact/text-gen/"}
epsilon_value=${epsilon_value:-8}
path_to_model=${path_to_model:-"/data/dp-fact/text-gen/models/princeton_wiki_DP_"}
enable_dp=${enable_dp:-true}
path_to_dataset=${path_to_dataset:-"/data/datasets/wikipedia-biographies-v1->-200.csv"}
epochs=${epochs:-5}
gradient_accumulation_steps=${gradient_accumulation_steps:-1}
#load_ckpt=${load_ckpt:-true}

if [ "$enable_dp" = false ]; then
  epsilon_value="inf"
fi
path_to_model=$path_to_model$epsilon_value
#TODO: Change the dataset name variable
python -m torch.distributed.run --nproc_per_node 4 lora_dp_trainer.py \
        --output_dir outputs \
        --model_name princeton \
        --path_to_dataset $path_to_dataset \
        --path_to_save $path_to_model \
        --enable_dp $enable_dp \
        --target_epsilon $epsilon_value \
        --target_delta 1e-5 \
        --dataset_name wiki \
        --save_total_limit 2 \
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

