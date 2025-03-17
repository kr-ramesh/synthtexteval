#!/bin/bash
#SBATCH --job-name="wiki-bio-gen"
#SBATCH --time=96:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:4
#SBATCH --partition=fastgpus
#SBATCH --output=test.txt

path_to_dataset="/data/datasets/wikipedia-biographies-v1-post-2020->-100.csv"
epochs=5
epsilon_value=8
sh train.sh "princeton-nlp/Sheared-LLaMA-1.3B" "/data/projects/syntheval/models/princeton_wiki_DP_" true $epsilon_value $path_to_dataset $epochs