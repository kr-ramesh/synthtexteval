
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
'''Train LLMs with DP using QLoRA'''

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
from load_models import load_model_tokenizer
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

logger = logging.getLogger(__name__)
    
@dataclass
class ModelArguments:
    model_name: str = field(default = "gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    dataset_name: str = field(default = "sst2", metadata={
        "help": "Dataset name in HuggingFace, e.g. 'sst2'"
    })
    path_to_save: str = field(default = "model_directory_default", metadata={
        "help": "Path to where the model weights are saved."
    })
    sequence_len: int = field(default = 1024, metadata={
        "help": "Maximum sequence length"
    })
    enable_dp: bool = field(default = False, metadata={
        "help": "Whether to enable Differentially Private Training"
    })
    path_to_dataset: str = field(default = "data1.csv", metadata={
        "help": "Path to the dataset."
    })
    path_to_load: str = field(default = "model_directory_default", metadata={
        "help": "Path to where the model weights are saved and need to be loaded from."
    })
    load_from_ckpt: bool = field(default = False, metadata={
        "help": "Load from ckpt"
    })
    
    
@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=32, metadata={
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


def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(args.model.model_name)

    # Load dataset
    dataset = data_utils.ALL_DATASETS[args.model.dataset_name](args, tokenizer)
    
    if dataset.classes is not None:
        target_max_len = dataset.target_max_len()
        logger.info(f"Labels tokenized into max length: {target_max_len}")

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset.dataset = dataset.dataset.map(
            dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset",
            remove_columns=dataset.dataset.column_names['train']
        )
    
    if args.lora.enable_lora:
        if not args.model.load_from_ckpt:
            logger.info("Using LoRA")
            peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode=False, r=args.lora.lora_dim, lora_alpha=args.lora.lora_alpha, lora_dropout=args.lora.lora_dropout)
            model = get_peft_model(model, peft_config)
        else:
            if(args.model.enable_dp):
                model.load_adapter(args.model.path_to_load + '_dir')
            else:
                model.load_adapter(args.model.path_to_load)
    else:
        logger.info("Not using LoRA")
    

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")
    
    def print_summary(result):
            print(f"Time: {result.metrics['train_runtime']:.2f}")
            print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
            print_gpu_utilization()

    model = model.to("cuda")
    print(model.num_parameters(only_trainable = True))
    
    if(args.model.enable_dp):
        print("Differentially Private Training: True")
        trainer = dp_utils.OpacusDPTrainer(
            args=train_args,
            model=model,
            train_dataset=dataset.dataset['train'],
            eval_dataset=dataset.dataset['validation'],
            tokenizer=tokenizer,
            compute_metrics=dataset.compute_metrics,
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
            privacy_args=privacy_args,
        )
        print("Trainer initialized.")
        if hasattr(trainer.model._module, "config"):
            # The following is for GradSampleModule wrapping
            ignore_keys = getattr(trainer.model._module.config, "keys_to_ignore_at_inference", [])
        elif hasattr(trainer.model._module.module, "config"):
            # The following is for GradSampleModule and DPDDP wrapping
            ignore_keys = getattr(trainer.model._module.module.config, "keys_to_ignore_at_inference", [])
        else:
            ignore_keys = []

        try:
            # A workaround to avoid the following error:
            # AttributeError: 'GradSampleModule' object has no attribute 'gradient_checkpointing_enable'
            # inside Trainer _inner_training_loop. Already done by prepare_model_for_kbit_training
            trainer.args.gradient_checkpointing = False
            result = trainer.train(ignore_keys_for_eval=ignore_keys)
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })

        privacy_engine = opacus.PrivacyEngine()
        privacy_engine.save_checkpoint(path = args.model.path_to_save, module = trainer.model, optimizer = trainer.optimizer)
        trainer.model._module.module.save_pretrained(args.model.path_to_save + '_dir')
    
    else:
        print("Differentially Private Training: False")
        trainer = Trainer(
            args=train_args,
            model=model,
            train_dataset=dataset.dataset['train'],
            eval_dataset=dataset.dataset['validation'],
            tokenizer=tokenizer,
            compute_metrics=dataset.compute_metrics,
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,)
        
        ignore_keys = []
        trainer.args.gradient_checkpointing = False
        result = trainer.train()
            
        trainer.save_model(args.model.path_to_save)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((argument_utils.TrainingArguments, argument_utils.PrivacyArguments, ModelArguments, LoraArguments))
    train_args, privacy_args, model_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, lora=lora_args))
