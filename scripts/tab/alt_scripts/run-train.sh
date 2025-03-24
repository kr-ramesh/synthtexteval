#!/bin/bash
#SBATCH --job-name="wiki-bio-gen"
#SBATCH --time=96:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:4
#SBATCH --partition=fastgpus
#SBATCH --output=test.txt

path_to_dataset="/home/kramesh3/syntheval/data/generator/data/tab"
dataset_name="tab"
epochs=5
epsilon_value=8
sh train.sh "princeton-nlp/Sheared-LLaMA-1.3B" "/home/kramesh3/syntheval/data/generator/models-test/" false $epsilon_value $dataset_name $path_to_dataset $epochs