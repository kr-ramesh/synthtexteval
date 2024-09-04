#Sample script for training models
foldername="princeton_mimic_10ICD_DP_8"
python icd_classification.py \
            --dataset_name "mimic" \
            --path_to_dataset "data-updated/$foldername/" \
            --is_train \
            --path_to_model "models-test/$foldername/" \
            --n_labels 10

#Sample script for testing models
baseline_path="baseline_downstream_data_10ICD"
python icd_classification.py 
            --dataset_name "mimic" \
            --path_to_dataset "data/$baseline_path/" \
            --is_test \
            --path_to_model "models/princeton_mimic_10ICD_DP_8" \
            --csv_output_path "outputs-from-testing.csv" \
            --n_labels 10