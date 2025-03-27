export model_name=$1
export dir_to_save_model=$2
export disable_dp=$3
export epsilon_value=$4
export dataset_name=$5
export path_to_dataset=$6
export epochs=$7
export gradient_accumulation_steps=$8
export load_ckpt=$9
export path_to_load_model=${10}
export enable_lora=${11}

model_name=${model_name:-"princeton-nlp/Sheared-LLaMA-1.3B"}
dir_to_save_model=${dir_to_save_model:-"./data/generator/models"}
disable_dp=${disable_dp:-true}
epsilon_value=${epsilon_value:-8}
dataset_name=${dataset_name:-"tab"}
path_to_dataset=${path_to_dataset:-"./data/generator/data/tab/"}
epochs=${epochs:-5}
gradient_accumulation_steps=${gradient_accumulation_steps:-1}
load_ckpt=${load_ckpt:-false}
path_to_load_model=${path_to_load_model:-None}
enable_lora=${enable_lora:-true}

if [ "$disable_dp" = true ]; then
  epsilon_value="inf"
fi
dir_to_save_model="${dir_to_save_model}/${model_name//\//_}_${dataset_name}_DP_${epsilon_value}"
echo "Directory to save the model into ${dir_to_save_model}"

#Enable dry_run True to test that it works
python -m torch.distributed.run --nproc_per_node 4 /home/kramesh3/synthtexteval/synthtexteval/generation/controllable/train_generator.py \
        --output_dir outputs \
        --dry_run True \
        --model_name $model_name \
        --path_to_dataset $path_to_dataset \
        --path_to_save_model $dir_to_save_model \
        --disable_dp $disable_dp \
        --target_epsilon $epsilon_value \
        --target_delta 1e-5 \
        --dataset_name $dataset_name \
        --save_total_limit 2 \
        --sequence_len 1024 \
        --per_device_train_batch_size 1 \
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
        --lora_dim 8 \
        --lora_alpha 8 \
        --lora_dropout 0.0 \
        --load_from_ckpt $load_ckpt \
        --path_to_load_model $path_to_load_model \
        --enable_lora $enable_lora \
        --target_modules "['q_proj', 'v_proj']" \
        --label_names labels \
        --gradient_checkpointing
