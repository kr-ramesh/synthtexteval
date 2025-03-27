import os
import opacus
import datasets
import transformers
import sys
import logging
import torch
import ast
import pandas as pd
import synthtexteval.generation.controllable.data_utils as data_utils
import synthtexteval.generation.controllable.argument_utils as argument_utils
import synthtexteval.generation.controllable.dp_utils as dp_utils
from synthtexteval.generation.controllable.load_models import load_model_tokenizer, load_dp_model
from tqdm import tqdm

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import Trainer, default_data_collator
from torch.utils.data import DataLoader

from pynvml import *

# Optional
os.environ['WANDB_DISABLED'] = "true"

#TODO: Change the argument variable names.

def inference(args: argument_utils.Arguments):
    """
    Inference function for the generator model to generate synthetic data via controllable text generation.
    """
    print(torch.cuda.current_device())
    transformers.set_seed(args.train.seed)
    
    # Load model and tokenizer
    args.train.do_train, args.train.do_eval = False, False
    model, tokenizer = load_model_tokenizer(args.model.model_name)

    # Load dataset
    dataset = data_utils.ALL_DATASETS[args.data.dataset_name](args, tokenizer)

    if dataset.classes is not None:
        target_max_len = dataset.target_max_len()

    if args.lora.enable_lora:
        print("Using LoRA")
        model.load_adapter(args.model.path_to_load_model)
    else:
        print("Not using LoRA")

    if args.train.local_rank == 0:
        print(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        print(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")
    
                           
    #if(not args.privacy.disable_dp):
        #print("Differentially Private Training: True")
        #model = load_dp_model(model, args.model.path_to_load_model + '_pvt')

    trainer = Trainer(
            args=args.train,
            model=model,
            tokenizer=tokenizer,
            compute_metrics=dataset.compute_metrics,
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,)

    df = dataset.compute_test_metrics(trainer, args.model)
    
    print("Saving results to file...")

    try:
        df.to_csv(args.model.path_to_save_test_output + '.csv')
    except:
        df.to_csv(args.model.path_to_load_model.replace('/', '_') + '_DP_' + str(args.privacy.disable_dp) + '_' + str(args.data.dataset_name) + '.csv', mode='w', index=False, header=False)
        
        
if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((argument_utils.TrainingArguments, argument_utils.PrivacyArguments, argument_utils.ModelArguments, argument_utils.DataArguments, argument_utils.LoraArguments))
    train_args, privacy_args, model_args, data_args, lora_args = arg_parser.parse_args_into_dataclasses()
    inference(argument_utils.Arguments(train=train_args, privacy=privacy_args, model=model_args, data = data_args, lora=lora_args))