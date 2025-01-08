from datasets import load_dataset, load_from_disk, Features, Value, ClassLabel, Dataset, DatasetDict
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch, ast, os
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score
from datasets import load_dataset
import os


#TODO: Need to remove some redundant parts of the code here. Add comments wherever necessary
#TODO: Add a dummy function and custom functionality for reading the dataset in case the user has their own format and preprocessing.

def read_data(data_dir, dataset_name, is_test=True):
    """
    Function to load and return the dataset for training or the evaluation of bias.

    Args:
        - data_dir (str): Path to the data directory.
        - dataset_name (str): Name of the dataset to be loaded. If an HF dataset with this name exists, it will be loaded.
        - is_test (bool): Whether to load test data only. If False, it loads train and validation sets.

    Returns:
        - dataset (DatasetDict or dict): A DatasetDict if an HF dataset is found, otherwise a dict of datasets loaded from CSV files.
    """
    print("Dataset Name: ", dataset_name)
    
    try:
        # Attempt to load the dataset from HF Hub
        dataset = load_dataset(dataset_name)
        print(f"Successfully loaded dataset '{dataset_name}' from Hugging Face Hub.")
    except ValueError:
        # Fall back to loading from local files
        if is_test:
            print("Loading test data")
            dataset = load_dataset('csv', data_files={"test": os.path.join(data_dir, "test.csv")})
        else:
            dataset = load_dataset('csv', data_files={
                "train": os.path.join(data_dir, "train.csv"),
                "validation": os.path.join(data_dir, "eval.csv")
            })
    
    return dataset

def load_model(model_name, path_to_model, n_labels, problem_type, ckpt_exists = False):

      """Loads the HuggingFace model for training/testing

       Args:
            - model_name (str): Name of the model to be loaded
            - path_to_model (str): Path to the model to be loaded, in case a checkpoint is provided

       Returns:
            - model : Returns the model
            - tokenizer : Returns the tokenizer corresponding to the model
      """
      if(problem_type == "mimic"):
          problem_type = "multi_label_classification"
      else:
          problem_type = "single_label_classification"
         
      if(ckpt_exists):
        print("Checkpoint exists: ", path_to_model, "\nLoading model from the checkpoint...")
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model, local_files_only=True, num_labels = n_labels, problem_type = problem_type, output_attentions = False, output_hidden_states = False,)
      else:
        print("Loading base model for fine-tuning...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = n_labels, problem_type = problem_type, output_attentions = False, output_hidden_states = False,)
      
      tokenizer = AutoTokenizer.from_pretrained(model_name)

      return model, tokenizer

def save_results_to_file(results, file_path):
    with open(file_path, 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

def evaluate_multilabel_classifier(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    hamming_loss_val = hamming_loss(y_true, y_pred)
    
    subset_accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'Precision': precision,
        'Recall': recall,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'Hamming Loss': hamming_loss_val,
        'Subset Accuracy': subset_accuracy,
    }


def tokenize_data(tokenizer, data, class_labels, dataset_name):

      input_ids, attention_masks = [], []

      for k, sent in enumerate(data):
          encoded_dict = tokenizer.encode_plus(str(sent), add_special_tokens = True, max_length = 512, truncation=True,
                                              pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt',)

          input_ids.append(encoded_dict['input_ids'])
          attention_masks.append(encoded_dict['attention_mask'])

      input_ids, attention_masks = torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)
      if(dataset_name == "mimic"):
         class_labels = [ast.literal_eval(item) for item in class_labels]
         num_labels = max([max(labels) for labels in class_labels]) + 1
         class_labels = torch.tensor([[1 if i in sub_list else 0 for i in range(num_labels)] for sub_list in class_labels])
         class_labels = class_labels.float()

      elif(dataset_name == "cps"):
        class_labels = torch.tensor(class_labels)

      return input_ids, class_labels, attention_masks
      

