#!/bin/bash
#SBATCH --job-name="inf-wiki"
#SBATCH --time=500:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=test-inf.txt

path_to_load_model="/home/kramesh3/syntheval/data/generator/models"
dataset_name="tab"
path_to_dataset="/home/kramesh3/syntheval/data/generator/data/tab/"
path_to_test_dataset=None
sh inf.sh "/home/kramesh3/syntheval/data/synthetic" "princeton-nlp/Sheared-LLaMA-1.3B" $path_to_load_model $dataset_name $path_to_dataset $path_to_test_dataset true 8
