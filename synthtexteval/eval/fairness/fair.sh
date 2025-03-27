path_to_output_predictions=$1
problem_type=$2
n_labels=$3
python eval.py \
    --path_to_output_predictions ${path_to_output_predictions} \
    --problem_type ${problem_type} \
    --num_labels ${n_labels}