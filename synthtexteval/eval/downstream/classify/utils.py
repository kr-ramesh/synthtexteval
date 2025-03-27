from datasets import load_dataset, load_from_disk, Features, Value, ClassLabel, Dataset, DatasetDict
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch, ast, os
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from datasets import load_dataset
import os


#TODO: Need to remove some redundant parts of the code here. Add comments wherever necessary
#TODO: Add a dummy function and custom functionality for reading the dataset in case the user has their own format and preprocessing.

def read_data(data_dir, is_test=True, is_synthetic = False):
    """
    Function to load and return the dataset for training.

    Args:
        - data_dir (str): Path to the data directory.
        - is_test (bool): Whether to load test data only. If False, it loads train and validation sets.
        - is_synthetic (bool): Whether to load synthetic data. If True, it loads the synthetic data from the data directory.

    Returns:
        - dataset (DatasetDict or dict): A DatasetDict if an HF dataset is found, otherwise a dict of datasets loaded from CSV files.
    """

    if os.path.exists(data_dir):
        # Loading from local files
        if is_test:
            print("Loading test data")
            if(data_dir.endswith(".csv")):
                dataset = load_dataset('csv', data_files={"test": data_dir})
            else:
                dataset = load_dataset('csv', data_files={"test": os.path.join(data_dir, "test.csv")})
        else:
            print("Loading training and validation data.")
            data_dict = {"train": os.path.join(data_dir, "train.csv"), "validation": os.path.join(data_dir, "validation.csv")}
            if(is_synthetic):
                print("Loading synthetic data from data directory")
                data_dict["synthetic"] = os.path.join(data_dir, "synthetic.csv")
            dataset = load_dataset('csv', data_files=data_dict)
    else:
        # Attempt to load the dataset from HF Hub or directory of the correct format
        dataset = load_dataset(data_dir)
        print(f"Successfully loaded dataset '{data_dir}' from Hugging Face Hub.")
    
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
         
      if(ckpt_exists):
        print("Checkpoint exists: ", path_to_model, "\nLoading model from the checkpoint...")
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model, local_files_only=True, num_labels = n_labels, problem_type = problem_type, output_attentions = False, output_hidden_states = False,)
      else:
        print("Loading base model for fine-tuning...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = n_labels, problem_type = problem_type, output_attentions = False, output_hidden_states = False,)
      
      tokenizer = AutoTokenizer.from_pretrained(model_name)

      return model, tokenizer

def save_results_to_file(results, file_path):
    # Save the results to a file
    with open(file_path, 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

def tokenize_data(tokenizer, data, class_labels, problem_type):
      """
      Tokenizes the input data and returns the input_ids, class_labels and attention_masks
      """
      
      input_ids, attention_masks = [], []

      for k, sent in enumerate(data):
          encoded_dict = tokenizer.encode_plus(str(sent), add_special_tokens = True, max_length = 512, truncation=True,
                                              pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt',)

          input_ids.append(encoded_dict['input_ids'])
          attention_masks.append(encoded_dict['attention_mask'])

      input_ids, attention_masks = torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)
      if(problem_type == "multi_label_classification"):
         class_labels = [ast.literal_eval(item) for item in class_labels]
         num_labels = max([max(labels) for labels in class_labels]) + 1
         class_labels = torch.tensor([[1 if i in sub_list else 0 for i in range(num_labels)] for sub_list in class_labels])
         class_labels = class_labels.float()

      elif(problem_type == "single_label_classification"):
        class_labels = torch.tensor(class_labels)

      return input_ids, class_labels, attention_masks


def experiment_compare_real_synthetic(args, retrain=False, plot_path='temp/results.png', metrics=['Precision', 'Recall', 'f1_micro']):
    """
    Function to compare performance on downstream task across real-only, synthetic-only, and synthetic augmentation training data settings.
    Generates plot to compare across scoring metrics.

    Args:
        - args (Arguments): training and model arguments.
        - retrain (bool): whether to retrain models with existing checkpoints (will retrain if true)
        - plot_path (str): where resulting plot is saved
        - metrics (list[str]): list of metrics to compare model performance (metrics must match headings in output csv headers)
    """
    
    model_args = args.model
    path_to_model = model_args.path_to_model
    model_args.is_train = True
    model_args.is_test = False
    for synth_arg in ['real-only', 'synthetic-train-only', 'synthetic-train-augment']:
        print(f"Training {synth_arg}:\n")
        model_args.synthetic_usage = synth_arg
        model_args.path_to_model = f'{path_to_model}/{synth_arg}'
        if not os.path.exists(model_args.path_to_model) or retrain:
            obj = Classifier(args = args)
            obj.finetune_model()
    
    model_args.is_train = False
    model_args.is_test = True
    if not os.path.exists(model_args.path_to_aggregated_results):
        for synth_arg in ['real-only', 'synthetic-train-only', 'synthetic-train-augment']:
            print(f"Testing {synth_arg}:\n")
            model_args.path_to_model = f'{path_to_model}/{synth_arg}'
            obj = Classifier(args = args)
            obj.test_model()
        
    if plot_path:
        eval_df = pd.read_csv(model_args.path_to_aggregated_results)
        plot_df = []
        for i, row in eval_df.iterrows():
            print(row['model_name'])
            name = row['model_name'][len(path_to_model)+1:]
            print(name)
            for metric in metrics:
                plot_df.append([name, metric, row[f'eval_{metric}']])
        plot_df = pd.DataFrame(plot_df, columns=['Training Data', 'Metric', 'Score'])
        plt.figure(figsize=(5,3))
        sns.barplot(plot_df, x='Metric', y='Score', hue='Training Data')
        plt.legend(loc='lower right')
        plt.savefig(plot_path)
