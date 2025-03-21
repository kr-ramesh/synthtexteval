import transformers
import sdgeval.generation.controllable.argument_utils as argument_utils
import pandas as pd
from sdgeval.generation.controllable.inference import inference
from sdgeval.generation.controllable.train_generator import train
from sdgeval.generation.controllable.testing_args import set_default_training_args, set_default_config_args


if __name__ == "__main__":

    train_args = set_default_training_args(dry_run=False)
    model_args, data_args = set_default_config_args()
    privacy_args, lora_args= argument_utils.PrivacyArguments(), argument_utils.LoraArguments()
    privacy_args.disable_dp = True
    
    data_args.dataset_name = 'tab'
    data_args.path_to_dataset = '../../data/generate/data/tab/'

    model_args.path_to_save_model = '../../data/generate/models/princeton_tab_noDP'
    
    train(argument_utils.Arguments(train=train_args, privacy=privacy_args, model=model_args, data = data_args, lora=lora_args))

    train_args = set_default_training_args(dry_run=False, dry_test_run = False)

    model_args.inference = True
    model_args.path_to_load_model = '../../data/generate/models/princeton_tab_noDP'
    model_args.num_return_seq = 5
    model_args.path_to_save_test_output = '../../data/synthetic/princeton_tab_synthetic_noDP.csv'
    
    inference(argument_utils.Arguments(train=train_args, privacy=privacy_args, model=model_args, data = data_args, lora=lora_args))