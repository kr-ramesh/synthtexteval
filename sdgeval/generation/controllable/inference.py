import os
import datasets
import transformers
import sys
import logging
import torch
import ast
import linear
import data_utils
import argument_utils
import dp_utils
import opacus
from load_models import SynthModels, load_dp_model
from tqdm import tqdm
import pandas as pd

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import Trainer, default_data_collator
from torch.utils.data import DataLoader

from pynvml import *

#TODO: Change the argument variable names.
#TODO: The LoRA arguments are at present hardcoded. This needs to change.
#TODO: Change the way the output is saved. Make sure the num_return_sequences is an argument accepted by the user.
#TODO: Change the function name from 'main' to something else

@dataclass
class ModelArguments:
    model_name: str = field(default = "gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    dataset_name: str = field(default = "sst2", metadata={
        "help": "Dataset name in HuggingFace, e.g. 'sst2'"
    })
    path_to_load: str = field(default = "model_directory_default", metadata={
        "help": "Path to where the model weights are saved and need to be loaded from."
    })
    save_output_path: str = field(default = None, metadata={
        "help": "Path to saved model output CSV file."
    })
    sequence_len: int = field(default = 128, metadata={
        "help": "Maximum sequence length"
    })
    enable_dp: bool = field(default = False, metadata={
        "help": "Whether or not the model is trained with DP"
    })
    path_to_dataset: str = field(default = "data1.csv", metadata={
        "help": "Path to the dataset."
    })
    path_to_test_dataset: str = field(default = "data1.csv", metadata={
        "help": "Path to the dataset for inference."
    })
    num_codes: int = field(default = 5, metadata={
        "help": "Number of control codes to sample from. 0 if you want all control codes."
    })
    num_samples: int = field(default = 500, metadata={
        "help": "Number of samples to take into account for training on a balanced dataset."
    })

    
@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    target_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        params["target_modules"] = ast.literal_eval(params["target_modules"][0])
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: argument_utils.TrainingArguments
    privacy: argument_utils.PrivacyArguments
    model: ModelArguments
    lora: LoraArguments

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def main(args: Arguments):
    #torch.cuda.init()
    #gpu_index = 1
    #torch.cuda.set_device(gpu_index)
    print(torch.cuda.current_device())
    transformers.set_seed(args.train.seed)
    
    # Load model and tokenizer
    obj = SynthModels(model_name = args.model.model_name, model_type = "causal")
    model = obj.model
    tokenizer = obj.tokenizer

    # Load dataset
    dataset = data_utils.ALL_DATASETS[args.model.dataset_name](tokenizer, args, inference = True)
                                                               #sequence_len = args.model.sequence_len, num_samples = args.model.num_samples, 
                                                            #path_to_dataset = args.model.path_to_dataset, num_codes = args.model.num_codes, inference = True)
    if dataset.classes is not None:
        target_max_len = dataset.target_max_len()

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing test dataset"):
        dataset.dataset = dataset.dataset.map(
            dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset",
            remove_columns=dataset.dataset.column_names['train']
        )


    if args.lora.enable_lora:
        print("Using LoRA")
        if(args.model.enable_dp):
            model.load_adapter(args.model.path_to_load + '_dir')
        else:
            model.load_adapter(args.model.path_to_load)
    else:
        print("Not using LoRA")

    if train_args.local_rank == 0:
        print(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        print(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")
    if(args.model.enable_dp):
        print("Differentially Private Training: True")
        model = load_dp_model(model, args.model.path_to_load)

        trainer = Trainer(
            args=train_args,
            model=model._module,
            train_dataset=dataset.dataset['train'],
            tokenizer=tokenizer,
            compute_metrics=dataset.compute_metrics,
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,)

        print("DP model has been loaded!")

        df = dataset.compute_test_metrics(trainer)
        print("Saving results to file!")
        try:
            df.to_csv(args.model.save_output_path + '.csv')
        except:
            df.to_csv(args.model.path_to_load.replace('/', '_') + '_DP_' + str(args.model.enable_dp) + '_' + str(args.model.dataset_name) +  '.csv', mode='w', index=False)
    
    else:
        print("Differentially Private Training: False")
        trainer = Trainer(
            args=train_args,
            model=model,
            train_dataset=dataset.dataset['train'],
            tokenizer=tokenizer,
            compute_metrics=dataset.compute_metrics,
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,)
        
        #trainer.model.from_pretrained(args.model.path_to_load)

        print("Model has been loaded!")
        
        df = dataset.compute_test_metrics(trainer)
        print("Saving results to file!")
        try:
            df.to_csv(args.model.save_output_path + '.csv')
        except:
            df.to_csv(args.model.path_to_load.replace('/', '_') + '_noDP_' + str(args.model.enable_dp) + '_' + str(args.model.dataset_name) + '.csv', mode='w', index=False, header=False)
        
        
if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((argument_utils.TrainingArguments, argument_utils.PrivacyArguments, ModelArguments, LoraArguments))
    train_args, privacy_args, model_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, lora=lora_args))