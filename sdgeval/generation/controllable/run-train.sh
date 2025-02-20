#!/bin/bash
#SBATCH --job-name="wiki-bio-gen"
#SBATCH --time=96:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:4
#SBATCH --partition=fastgpus
#SBATCH --output=text-outputs/train-model-wikibio-updated-30e.txt

#sh train.sh "/data/psd/cps-updated/" "/data/models-exp/princeton_cps_T1_DP_" false "inf" 
#sh train.sh "/data/dp-fact/text-gen/" "/data/dp-fact/text-gen/models/princeton_wiki_DP_" false "inf" 


#sh train.sh "/data/dp-fact/text-gen/" "/data/dp-fact/text-gen/models/princeton_wiki_updated_DP_" false "inf" $path_to_dataset 5 1

path_to_dataset="/data/datasets/wikipedia-biographies-v1-post-2020->-100.csv"
sh train.sh "/data/dp-fact/text-gen/" "/data/dp-fact/text-gen/models/princeton_wiki_updated_DP_e30_" false "inf" $path_to_dataset 15 1

sh train.sh "/data/dp-fact/text-gen/" "/data/dp-fact/text-gen/models/princeton_wiki_updated_DP_e30_" true 8 $path_to_dataset 30 64

sh train.sh "/data/dp-fact/text-gen/" "/data/dp-fact/text-gen/models/princeton_wiki_updated_DP_e30_" true 4 $path_to_dataset 30 64