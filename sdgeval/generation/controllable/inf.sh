export output_csv_path=$1
export model_type=$2
export path_to_model=$3
export path_to_test_dataset=$4
export enable_dp=$5
export epsilon_value=$6

output_csv_path=${output_csv_path:-"princeton_wiki_DP_8_outputs.csv"}
model_type=${model_type:-"princeton"}
path_to_model=${path_to_model:-"/data/dp-fact/text-gen/models/princeton_wiki_DP_"}
path_to_test_dataset=${path_to_test_dataset:-"/data/dp-fact/text-gen/models/princeton_wiki_data/eval.csv"}
epsilon_value=${epsilon_value:-8}
enable_dp=${enable_dp:-true}


if [ "$enable_dp" = false ]; then
  epsilon_value="inf"
fi
path_to_model=$path_to_model$epsilon_value
path_to_dataset="/data/dp-fact/text-gen/models/princeton_wiki_updated_data/"
echo $path_to_test_dataset
echo $path_to_model

python inference.py \
        --output_dir outputs \
        --enable_dp ${enable_dp} \
        --model_name "${model_type}" \
        --path_to_load "${path_to_model}" \
        --path_to_dataset "${path_to_dataset}" \
        --path_to_test_dataset ${path_to_test_dataset} \
        --save_output_path "${output_csv_path}" \
        --dataset_name wiki \
        --target_epsilon ${epsilon_value} \
        --sequence_len 1024 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy steps \
        --eval_steps 10 \
        --save_strategy no \
        --log_level info \
        --per_device_eval_batch_size 4 \
        --eval_accumulation_steps 1 \
        --seed 42 \
        --target_delta 1e-5 \
        --per_sample_max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --remove_unused_columns False \
        --num_train_epochs 5 \
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
        --gradient_checkpointing