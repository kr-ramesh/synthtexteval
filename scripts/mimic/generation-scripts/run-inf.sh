#!/bin/bash
path_to_load_model="./data/generator/models"
dataset_name="mimic"
path_to_dataset="./data/generator/data/mimic/mimic-iii-3"
path_to_test_dataset=None
sh inf.sh "./data/data" "princeton-nlp/Sheared-LLaMA-1.3B" $path_to_load_model $dataset_name $path_to_dataset $path_to_test_dataset false 8
