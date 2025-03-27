#!/bin/bash
# Define the path to the folder TAB has been downloaded
PATH_TO_DATA_FOLDER=$1
PATH_TO_DATA_FOLDER=${PATH_TO_DATA_FOLDER:-"./synthtexteval/"}
# This script uses pre-defined arguments. For the scripts with modifiable arguments, refer to the synthtexteval.generation.controllable directory.
# Change the privacy_args.disable_dp in the train_and_gen_tab.py to train a differentially private model
python -m torch.distributed.run --nproc_per_node 4 train.py $PATH_TO_DATA_FOLDER
export CUDA_VISIBLE_DEVICES=0
python gen.py $PATH_TO_DATA_FOLDER