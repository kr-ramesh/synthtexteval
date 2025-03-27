export synth_data_file=$1
export sample_size=$2
export test_file=$3
export num_train_epochs=$4


synth_data_file=${synth_data_file:-"/data/coref-updated-files/filtered/mimic/princeton_mimic_10ICD_DP_8.csv"}
sample_size=${sample_size:-1000}
test_file=${test_file:-"/home/ngandh17/coref/s2e-coref/data/test.i2b2.jsonlines"}
num_train_epochs=${num_train_epochs:-10}

# download the base pretrained model
temp_output_dir="./temp"
model_dir=$temp_output_dir/base_pretrained_model
export predict_file=$temp_output_dir/silver.jsonlines
export predict_file_write=$temp_output_dir/silver.pred.jsonlines

# download pretrained coref model
#mkdir -p $temp_output_dir
#curl -L https://www.dropbox.com/sh/7hpw662xylbmi5o/AAC3nfP4xdGAkf0UkFGzAbrja?dl=1 > temp_model.zip
#unzip temp_model.zip -d $model_dir
#rm -rf temp_model.zip

# set up synthetic data
python minimize_synth.py $synth_data_file $temp_output_dir $sample_size

# report performance difference
python run_coref_comparison.py \
	--output_dir=$temp_output_dir \
        --model_type=longformer \
        --base_model_name_or_path=$model_dir \
        --tokenizer_name=allenai/longformer-large-4096 \
        --test_file=$test_file \
        --do_infer \
        --num_train_epochs=$num_train_epochs \
        --logging_steps=100 \
        --save_steps=1000 \
        --eval_steps=150 \
        --max_seq_length=4000 \
        --predict_file=$predict_file \
        --predict_file_write=$predict_file_write \
        --normalise_loss \
        --max_total_seq_len=4000 \
        --experiment_name=eval_model \
        --warmup_steps=5600 \
        --adam_epsilon=1e-6 \
        --head_learning_rate=3e-4 \
        --learning_rate=1e-5 \
        --adam_beta2=0.98 \
        --weight_decay=0.01 \
        --dropout_prob=0.3 \
        --save_if_best \
        --top_lambda=0.4  \
        --tensorboard_dir=$temp_output_dir/tb \

# clean up
#rm -rf $temp_output_dir