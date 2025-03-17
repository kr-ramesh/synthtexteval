#!/bin/bash
#SBATCH --job-name="inf-wiki"
#SBATCH --time=500:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=test.txt

path_to_load_model="/data/dp-fact/text-gen/models/princeton_wiki_test_DP_"
path_to_test_dataset="/data/projects/syntheval/models/princeton_wiki_data/eval.csv"
sh inf.sh "princeton_wiki_DP_8_outputs.csv" "princeton-nlp/Sheared-LLaMA-1.3B" $path_to_load_model $path_to_test_dataset true 8

#epsilon_value=inf
#enable_dp=false
#sh inf.sh "princeton_wiki_DP_inf_e30_outputs" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value
