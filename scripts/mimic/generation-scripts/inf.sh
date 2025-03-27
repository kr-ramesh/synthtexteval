export dir_to_save_test_output=$1
export model_name=$2
export dir_to_load_model=$3
export dataset_name=$4
export path_to_dataset=$5
export path_to_test_dataset=$6
export disable_dp=$7
export epsilon_value=$8
export enable_lora=$9
export num_return_seq=${10}

dir_to_save_test_output=${dir_to_save_test_output:-"./data/data/"}
model_name=${model_name:-"princeton-nlp/Sheared-LLaMA-1.3B"}
dir_to_load_model=${dir_to_load_model:-"./data/generator/models/"}
dataset_name=${dataset_name:-"tab"}
#Either specify path_to_dataset or path_to_test_dataset
path_to_dataset=${path_to_dataset:-"./data/generator/tab/"}
path_to_test_dataset=${path_to_test_dataset:-"./data/generator/tab/"}
epsilon_value=${epsilon_value:-8}
disable_dp=${disable_dp:-true}
enable_lora=${enable_lora:-true}
num_return_seq=${num_return_seq:-2}

if [ "$disable_dp" = true ]; then
  epsilon_value="inf"
fi

path_to_load_model="${dir_to_load_model}/${model_name//\//_}_${dataset_name}_DP_${epsilon_value}"
path_to_save_test_output="${dir_to_save_test_output}/${model_name//\//_}_${dataset_name}_DP_${epsilon_value}_outputs"

echo $path_to_test_dataset
echo $path_to_model
#Enable dry_test_run True to test that it works
python /home/kramesh3/synthtexteval/synthtexteval/generation/controllable/inference.py \
        --dry_test_run True \
        --output_dir outputs \
        --disable_dp ${disable_dp} \
        --inference True \
        --model_name "${model_name}" \
        --dry_test_run True \
        --path_to_load_model "${path_to_load_model}" \
        --path_to_dataset "${path_to_dataset}" \
        --path_to_test_dataset ${path_to_test_dataset} \
        --path_to_save_test_output "${path_to_save_test_output}" \
        --dataset_name ${dataset_name} \
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
        --enable_lora $enable_lora \
        --target_modules "['q_proj', 'v_proj']" \
        --label_names labels \
        --gradient_checkpointing