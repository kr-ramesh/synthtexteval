import sys
import transformers
import synthtexteval.generation.controllable.argument_utils as argument_utils
import pandas as pd
from synthtexteval.generation.controllable.inference import inference
from synthtexteval.generation.controllable.train_generator import train
from synthtexteval.generation.controllable.testing_args import set_default_training_args, set_default_config_args


if __name__ == "__main__":
    # Training
    model_args, data_args = set_default_config_args()
    privacy_args, lora_args= argument_utils.PrivacyArguments(), argument_utils.LoraArguments()
    privacy_args.disable_dp = False
    
    if(privacy_args.disable_dp):
        model_path = f'{model_args.model_name.replace("/", "_")}_tab_DP_inf'
    else:
        model_path = f'{model_args.model_name.replace("/", "_")}_tab_DP_{privacy_args.target_epsilon}'

    data_args.dataset_name = 'tab'
    data_args.path_to_dataset = f'{sys.argv[1]}/data/generator/data/tab/'
    # Inference
    
    train_args = set_default_training_args(dry_run=False, dry_test_run = False)
    model_args.inference = True
    model_args.path_to_load_model = f'{sys.argv[1]}/data/generator/models/{model_path}'
    
    model_args.num_return_seq = 2
    model_args.path_to_save_test_output = f'{sys.argv[1]}/data/synthetic/{model_path}_outputs-final'
    
    inference(argument_utils.Arguments(train=train_args, privacy=privacy_args, model=model_args, data = data_args, lora=lora_args))