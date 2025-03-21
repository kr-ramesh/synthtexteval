#!/bin/bash
#SBATCH --job-name="train-classifier"
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=test.out

sh train.sh bert-base-uncased "sst2" temp/bert-base-uncased-sst2-v1 2 True False