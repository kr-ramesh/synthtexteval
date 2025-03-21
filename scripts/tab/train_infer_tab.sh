#!/bin/bash
#SBATCH --job-name="tab-gen"
#SBATCH --time=96:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:4
#SBATCH --partition=fastgpus
#SBATCH --output=tab-training.txt

python train_infer_tab.py