import sys
import transformers
import synthtexteval.generation.controllable.argument_utils as argument_utils
import pandas as pd
from synthtexteval.generation.controllable.inference import inference
from synthtexteval.generation.controllable.train_generator import train
from synthtexteval.generation.controllable.testing_args import set_default_training_args, set_default_config_args


if __name__ == "__main__":
    # Training
    train_args = set_default_training_args(dry_run=False)
    model_args, data_args = set_default_config_args()
    privacy_args, lora_args= argument_utils.PrivacyArguments(), argument_utils.LoraArguments()
    privacy_args.disable_dp = False

    if(privacy_args.disable_dp):
        print("Non-differentially private training")
        train_args.gradient_accumulation_steps = 1
        model_path = f'{model_args.model_name.replace("/", "_")}_tab_DP_inf'
        train_args.num_train_epochs = 5
    else:
        print("Differentially private training...")
        model_path = f'{model_args.model_name.replace("/", "_")}_tab_DP_{privacy_args.target_epsilon}'
        train_args.gradient_accumulation_steps = 16
        train_args.num_train_epochs = 50
    
    train_args.learning_rate = 3e-3
    model_args.path_to_save_model = f'{sys.argv[1]}/data/generator/models/{model_path}'
            
    data_args.dataset_name = 'tab'
    data_args.path_to_dataset = f'{sys.argv[1]}/data/generator/data/tab/'
        
    print(f"Path the model is saved to : {model_args.path_to_save_model}")

    train(argument_utils.Arguments(train=train_args, privacy=privacy_args, model=model_args, data = data_args, lora=lora_args))