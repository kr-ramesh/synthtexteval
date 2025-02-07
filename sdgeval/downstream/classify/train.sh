export model_name=$1
export dataset_name=$2
export path_to_model=$3
export path_to_dataset=$4
export num_labels=$5
export is_train=$6
export is_test=$7

# Default values if arguments are not provided
model_name=${model_name:-"bert-base-uncased"}
dataset_name=${dataset_name:-"stanfordnlp/sst2"}
path_to_model=${path_to_model:-"temp/bert-base-uncased-sst2-trial"}
path_to_dataset=${path_to_dataset:-"/home/umd-dsmolyak/scr4_afield6/umd-dsmolyak/sdg-eval-scr/data/sst2"}
num_labels=${num_labels:-2}
is_train=${is_train:-false}
is_test=${is_test:-true}


python train_classifier.py \
                --model_name ${model_name} \
                --dataset_name ${dataset_name} \
                --is_train ${is_train} \
                --is_test ${is_test} \
                --path_to_model ${path_to_model} \
                --path_to_dataset ${path_to_dataset} \
                --n_labels ${num_labels} \
                --max_steps 10 \
                --text_field 'sentence' \
                --label_field 'label'