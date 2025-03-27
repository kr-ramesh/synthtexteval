import os
import datasets
import transformers
import sys
import logging
import torch
import ast
import synthtexteval.generation.controllable.data_utils as data_utils
import synthtexteval.generation.controllable.argument_utils as argument_utils
import synthtexteval.generation.controllable.dp_utils as dp_utils
import opacus
from synthtexteval.generation.controllable.load_models import load_model_tokenizer
from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from transformers import Trainer

from pynvml import *

# Optional
os.environ['WANDB_DISABLED'] = "true"

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

logger = logging.getLogger(__name__)

def train(args: argument_utils.Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.train.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.train.local_rank}, device: {args.train.device}, n_gpu: {args.train.n_gpu}, "
        f"distributed training: {bool(args.train.local_rank != -1)}, 16-bits training: {args.train.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args.train}")
    logger.info(f"Privacy parameters {args.privacy}")

    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(args.model.model_name)

    # Load dataset
    dataset = data_utils.ALL_DATASETS[args.data.dataset_name](args, tokenizer)
    
    if dataset.classes is not None:
        target_max_len = dataset.target_max_len()
        logger.info(f"Labels tokenized into max length: {target_max_len}")

    # Tokenize data
    with args.train.main_process_first(desc="tokenizing dataset"):
        dataset.dataset = dataset.dataset.map(
            dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset",
            remove_columns=dataset.dataset.column_names['train'],
            load_from_cache_file=False
        )
    
    if args.lora.enable_lora:
        if not args.model.load_from_ckpt:
            logger.info("Using LoRA")
            peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode=False, r=args.lora.lora_dim, lora_alpha=args.lora.lora_alpha, lora_dropout=args.lora.lora_dropout)
            model = get_peft_model(model, peft_config)
        else:
            print("Loading from a pretrained checkpoint...")
            model.load_adapter(args.model.path_to_load_model)
    else:
        logger.info("Not using LoRA")
    

    if args.train.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")
    
    def print_summary(result):
            print(f"Time: {result.metrics['train_runtime']:.2f}")
            print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
            print_gpu_utilization()

    model = model.to("cuda")
    print(model.num_parameters(only_trainable = True))
    
    if(not args.privacy.disable_dp):
        print("Differentially Private Training: True")
        trainer = dp_utils.OpacusDPTrainer(
            args=args.train,
            model=model,
            train_dataset=dataset.dataset['train'],
            tokenizer=tokenizer,
            compute_metrics=dataset.compute_metrics,
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
            privacy_args=args.privacy,
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
        privacy_engine.save_checkpoint(path = args.model.path_to_save_model + '_pvt', module = trainer.model, optimizer = trainer.optimizer)
        try:
            trainer.model._module.save_pretrained(args.model.path_to_save_model)
        except:
            trainer.model._module.module.save_pretrained(args.model.path_to_save_model)
    
    else:
        print("Differentially Private Training: False")
        trainer = Trainer(
            args=args.train,
            model=model,
            train_dataset=dataset.dataset['train'],
            tokenizer=tokenizer,
            compute_metrics=dataset.compute_metrics,
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,)
        
        ignore_keys = []
        trainer.args.gradient_checkpointing = False
        result = trainer.train()
            
        trainer.save_model(args.model.path_to_save_model)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((argument_utils.TrainingArguments, argument_utils.PrivacyArguments, argument_utils.ModelArguments, argument_utils.DataArguments, argument_utils.LoraArguments))
    train_args, privacy_args, model_args, data_args, lora_args = arg_parser.parse_args_into_dataclasses()
    train(argument_utils.Arguments(train=train_args, privacy=privacy_args, model=model_args, data = data_args, lora=lora_args))
