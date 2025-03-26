#!/bin/bash

path_to_dataset="./data/generator/data/tab"
dataset_name="tab"
epochs=5
epsilon_value=8
sh train.sh "princeton-nlp/Sheared-LLaMA-1.3B" "./data/generator/models/" false $epsilon_value $dataset_name $path_to_dataset $epochs