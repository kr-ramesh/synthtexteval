#!/bin/bash
#SBATCH --job-name="inf-wiki"
#SBATCH --time=500:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=outputs/inference-updated-dp-30e.txt

model_type="princeton" 
path_to_model="/data/dp-fact/text-gen/models/princeton_wiki_updated_DP_e30_"
path_to_test_dataset="/data/dp-fact/text-gen/models/princeton_wiki_updated_data/inf.csv"

epsilon_value=inf
enable_dp=false
sh inf.sh "princeton_wiki_DP_inf_e30_outputs" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value

epsilon_value=8
enable_dp=true
sh inf.sh "princeton_wiki_DP_8_e30_outputs" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value

epsilon_value=4
enable_dp=true
sh inf.sh "princeton_wiki_DP_4_e30_outputs" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value

#epsilon_value=inf
#enable_dp=false
#sh inf.sh "princeton_wiki_DP_inf_outputs" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value


#path_to_test_dataset="/data/dp-fact/text-gen/inf/wiki-pretraining-random-1000.csv"
#epsilon_value=8
#enable_dp=true
#sh inf.sh "princeton_wiki_DP_8_outputs_1000_random" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value

#epsilon_value=4
#sh inf.sh "princeton_wiki_DP_4_outputs_1000_random" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value

#epsilon_value=inf
#enable_dp=false
#sh inf.sh "princeton_wiki_DP_inf_outputs_1000_random" $model_type $path_to_model $path_to_test_dataset $enable_dp $epsilon_value