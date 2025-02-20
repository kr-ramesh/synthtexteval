
import transformers
import pandas as pd
import torch
import opacus
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import psutil
dict_models = {'princeton': 'princeton-nlp/Sheared-LLaMA-1.3B',
               'eleuther_neo': 'EleutherAI/gpt-neo-1.3B',
               'microsoft_phi': 'microsoft/phi-1_5',
               'gpt2':'gpt2', 
               'princeton_2': 'princeton-nlp/Sheared-LLaMA-2.7B'}

dict_attention_models = {'princeton' : False, 'eleuther_neo': True, 'microsoft_phi': False, 'gpt2': True, 'princeton_2' : False}

#TODO: This entire file needs cleaning. No need to accept only models that are present in dict_models - instead, both of the arguments for attn and the model name can be fed in by the user.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', choices=['princeton', 'eleuther_neo', 'microsoft_phi', 'gpt2', 'princeton_2'], type=str,
                        help='The model used to generate synthetic data.')
    parser.add_argument('--tokenizer_name', required=False, default = None, type=str,
                        help='The tokenizer for the model used to generate synthetic data.')
    parser.add_argument('--model_type', required=True, type=str,
                        help='The type of the model used to generate synthetic data.')
    parser.add_argument('--prompt', required=False, type=str,
                        help='The type of prompt to be fed into the model.')
    parser.add_argument('--iterations', default = 3, required=False, type=int,
                        help='The number of iterations to run inference over the model for.')
    parser.add_argument('--file_name', default='outputs', required=False, type=str,
                        help='The name of the file to save results into.')
    parser.add_argument('--prompt_filepath', default='prompts.txt', required=False, type=str,
                        help='The name of the file to read the list of prompts from.')

    args = parser.parse_args()

    return args


class SynthModels:

    def __init__(self, model_name, model_type, prompt = None):

        self.model_name = model_name
        self.model_type = model_type
        self.prompt = prompt

        print("CPU: ", psutil.cpu_percent(4))
        print("Model: ", dict_models[self.model_name])

        if self.model_type == 'causal':

            self.model = AutoModelForCausalLM.from_pretrained(dict_models[self.model_name])
            self.tokenizer = AutoTokenizer.from_pretrained(dict_models[self.model_name])

            if(self.model_name == 'eleuther_neo'):
                self.tokenizer.pad_token = self.tokenizer.eos_token

            else:

                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

        elif self.model_type == 'microsoft_phi':

            self.model = AutoModelForCausalLM.from_pretrained(dict_models[self.model_name], trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(dict_models[self.model_name], trust_remote_code=True)
        
        #self.tokenizer.add_special_tokens({'Diagnosis Information': '<i>'}) <d>
    def model_inference(self, input_text):

        if self.prompt:
            input_text = self.add_prompt(input_text)

        inputs = self.tokenizer(input_text, return_tensors="pt", return_attention_mask = dict_attention_models[self.model_name], return_token_type_ids = False, padding = True, truncation = True)

        print("Outputs:")
        outputs = self.model.generate(**inputs, do_sample=True, max_length = 300, top_k=50, top_p=0.95, num_return_sequences=3)
        text = self.tokenizer.batch_decode(outputs)

        return text[0]

    def add_prompt(self, input_text):
        return prompt_dict[self.prompt] + input_text


def load_dp_model(model, path_to_load):
    transformers.set_seed(42)
    
    # Load model and tokenizer
    model.train()
    
    privacy_engine = opacus.PrivacyEngine()
    model = privacy_engine._prepare_model(model)
    
    checkpoint = torch.load(path_to_load, {})
    module_load_dict_kwargs = {'strict': False}
    model.load_state_dict(checkpoint["module_state_dict"], **(module_load_dict_kwargs or {}))
    
    print("Differentially Private Model has been loaded!")

    return model