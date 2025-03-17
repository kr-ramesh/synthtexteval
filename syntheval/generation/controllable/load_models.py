
import transformers
import pandas as pd
import torch
import opacus
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import psutil
               
def load_model_tokenizer(model_name):
    """
    Load model and tokenizer from HuggingFace.
    """
    print("CPU: ", psutil.cpu_percent(4))
    print("Model: ", model_name)

    if model_name == 'microsoft/phi-1_5':
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if(model_name == 'EleutherAI/gpt-neo-1.3B'):
            tokenizer.pad_token = tokenizer.eos_token

        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def load_dp_model(model, path_to_load):
    transformers.set_seed(42)
    
    # Load model and tokenizer
    model.train()
    
    privacy_engine = opacus.PrivacyEngine()
    model = privacy_engine._prepare_model(model)
    
    checkpoint = torch.load(path_to_load, {}, weights_only = False)
    module_load_dict_kwargs = {'strict': False}
    model.load_state_dict(checkpoint["module_state_dict"], **(module_load_dict_kwargs or {}))
    
    print("Differentially Private Model has been loaded!")

    return model