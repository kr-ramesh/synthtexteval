export path_to_save_test_output=$1
export model_name=$2
export path_to_load_model=$3
export path_to_test_dataset=$4
export disable_dp=$5
export epsilon_value=$6

path_to_save_test_output=${path_to_save_test_output:-"princeton_wiki_DP_8_outputs.csv"}
model_name=${model_name:-"princeton-nlp/Sheared-LLaMA-1.3B"}
path_to_load_model=${path_to_load_model:-"/data/dp-fact/text-gen/models/princeton_wiki_DP_"}
path_to_test_dataset=${path_to_test_dataset:-"/data/projects/sdgeval/models/princeton_wiki_data/eval.csv"}
epsilon_value=${epsilon_value:-8}
disable_dp=${enable_dp:-true}

if [ "$disable_dp" = false ]; then
  epsilon_value="inf"
fi

path_to_load_model=$path_to_load_model$epsilon_value
path_to_dataset="/data/dp-fact/text-gen/models/princeton_wiki_updated_data/"
echo $path_to_test_dataset
echo $path_to_model

#Enable dry_test_run True to test that it works
python inference.py \
        --output_dir outputs \
        --disable_dp ${disable_dp} \
        --inference True \
        --model_name "${model_name}" \
        --dry_test_run True \
        --path_to_load_model "${path_to_load_model}" \
        --path_to_dataset "${path_to_dataset}" \
        --path_to_test_dataset ${path_to_test_dataset} \
        --path_to_save_test_output "${path_to_save_test_output}" \
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
        --lora_dim 8 \
        --lora_alpha 8 \
        --lora_dropout 0.0 \
        --enable_lora \
        --target_modules "['q_proj', 'v_proj']" \
        --label_names labels \
        --gradient_checkpointing