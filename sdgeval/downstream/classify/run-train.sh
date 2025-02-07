#!/bin/bash
#SBATCH --job-name="test-classifier"
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --output=temp/testing-classification.txt

source ../../.env/bin/activate

sh train.sh bert-base-uncased stanfordnlp/sst2 temp/bert-base-uncased-sst2-trial /home/umd-dsmolyak/scr4_afield6/umd-dsmolyak/sdg-eval-scr/data/sst2 2 False True
