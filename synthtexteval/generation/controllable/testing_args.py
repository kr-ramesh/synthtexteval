
"""
This file provides utils for setting default arguments for training and testing. 
"""

from synthtexteval.generation.controllable.argument_utils import TrainingArguments, PrivacyArguments, ModelArguments, DataArguments

def set_default_config_args():
    model_args = ModelArguments(model_name = "princeton-nlp/Sheared-LLaMA-1.3B")
    data_args = DataArguments(dataset_name="hfhub",
            path_to_dataset="fancyzhx/ag_news")

    return model_args, data_args

def set_default_config_test_args():
    model_args = ModelArguments(model_name = "princeton-nlp/Sheared-LLaMA-1.3B", )
    data_args = DataArguments(dataset_name="hfhub",
            path_to_dataset="fancyzhx/ag_news")

    return model_args, data_args

def set_default_training_args(dry_run = False, dry_test_run = False):
    return TrainingArguments(output_dir = "/data/projects/synthtexteval/models/",
        evaluation_strategy = "no",
        num_train_epochs = 3,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 8,
        save_strategy = "steps",
        save_total_limit = 2,
        weight_decay=0.01,
        remove_unused_columns=False,
        logging_steps=4,
        learning_rate=3e-4,
        disable_tqdm=True,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        lr_scheduler_type="constant",
        max_grad_norm=0.0,
        eval_steps = None,
        log_level='info',
        seed = 42,
        dry_run = dry_run,
        dry_test_run = dry_test_run)
    

def set_default_privacy_args():
    return PrivacyArguments(target_epsilon = 8,
            target_delta = 1e-5,
            per_sample_max_grad_norm=1.0,)