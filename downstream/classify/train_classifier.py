from load_utils import read_data, load_model, tokenize_data, evaluate_multilabel_classifier
from datasets import concatenate_datasets
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Any, Callable, List, Optional, Union, Dict, Sequence
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from transformers import Trainer
from transformers import TrainingArguments as HfTrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import transformers
import argparse, torch
import numpy as np
import pandas as pd
import os

os.environ["WANDB_DISABLED"] = "true"

#TODO: Comment the code
#TODO: Shift around function classes, make the code look neater
#TODO: Change test function so that in cases where there is no ground truth available, you can just save the predictions

@dataclass
class TrainingArguments(HfTrainingArguments):
    output_dir: str = field(default="top3")
    evaluation_strategy: str = field(default="epoch")
    num_train_epochs: int = field(default=3)
    metric_for_best_model: str = field(default="f1")
    per_device_eval_batch_size: int = field(default=8)
    save_strategy: str = field(default="steps")
    save_total_limit: int = field(default=2)

    def __post_init__(self):
        super().__post_init__()    

@dataclass
class MiscArguments:
    model_name: str = field(default="bert-base-uncased", metadata={
        "help": "Path to the model to be used for training or testing. Can be a HuggingFace model or path to an existing fine-tuned model."
    })
    dataset_name: str = field(default="mimic", metadata={
        "help": "Name of the dataset"
    })
    path_to_dataset: Optional[str] = field(default="", metadata={
        "help": "Path to dataset directory to be used for all data."
    })
    # path_to_train_dataset: Optional[str] = field(default="", metadata={
    #     "help": "Path to dataset directory to be used for training."
    # })
    # path_to_test_dataset: Optional[str] = field(default="", metadata={
    #     "help": "Path to dataset directory to be used for testing."
    # })
    path_to_model: str = field(default="N/A", metadata={
        "help": "Path to HuggingFace model to be trained"
    })
    path_to_output_csv: str = field(default="outputs.csv", metadata={
        "help": "Path to where the downstream outputs are saved"
    })
    n_labels: int = field(default=10, metadata={
        "help": "Number of classes in the dataset."
    })
    problem_type: str = field(default="single_label_classification", metadata={
        "help": "Specify whether it is multiclass or multilabel classification."
    })
    path_to_aggregated_results: str = field(default="inf-aggs.csv", metadata={
        "help": "Path to where the downstream metric results for a given model over the test data is saved."
    })
    synthetic_usage: Optional[str] = field(default="", metadata={
        "help": "Whether and how synthetic data should be used (train augmentation, train synthetic-only, testing)"
    })
    text_field: str = field(default="text")
    label_field: str = field(default="label")
    is_train: Optional[bool] = field(default = False)
    is_test: Optional[bool] = field(default = False)

@dataclass
class Arguments:
    train: TrainingArguments
    model: MiscArguments

def save_predictions(Y_pred, Y_true, path_to_save):
    print("Saving file!")
    df = pd.DataFrame({'Predicted': Y_pred, 'Ground': Y_true})
    df.to_csv(path_to_save)

class ModelFT():

    def __init__(self, args):

        self.model_args = args.model
        self.training_args = args.train

        if(self.model_args.is_train):
            self.dataset = read_data(data_dir = self.model_args.path_to_dataset, dataset_name = self.model_args.dataset_name, is_test = False)
            self.model, self.tokenizer = load_model(model_name = self.model_args.model_name, path_to_model = self.model_args.path_to_model, problem_type = self.model_args.problem_type, n_labels = self.model_args.n_labels)
        elif(self.model_args.is_test):
            self.dataset = read_data(data_dir = self.model_args.path_to_dataset, dataset_name = self.model_args.dataset_name, is_test = True)
            self.dataset = self.dataset['test']
            if self.model_args.synthetic_usage == 'synthetic-evaluation':
                self.dataset = self.dataset['synthetic']
            self.model, self.tokenizer = load_model(model_name = self.model_args.model_name, path_to_model = self.model_args.path_to_model, problem_type = self.model_args.problem_type, ckpt_exists = True, n_labels = self.model_args.n_labels)
        
        self.device = "cuda"


    def process_map(self, data):
        processed_datasets = data.map(self.preprocess_function,
                                              batched=True,
                                              num_proc=1,
                                              load_from_cache_file=False,
                                              desc="Running tokenizer on dataset",)

        processed_datasets = processed_datasets.remove_columns(data.column_names)

        return processed_datasets

    def preprocess_function(self, examples):

        train_input_ids, train_class_labels, train_attention_masks = tokenize_data(self.tokenizer, examples[self.model_args.text_field], examples[self.model_args.label_field], self.model_args.dataset_name)
        examples['input_ids'] = train_input_ids
        examples['labels'] = train_class_labels
        examples['attention_mask'] = train_attention_masks

        return examples

    def compute_metrics_multiclass(self, eval_pred):
        logits, ytrue = eval_pred
        ypred = np.argmax(logits, axis=-1)
        precision = precision_score(ytrue, ypred, average='macro')
        recall = recall_score(ytrue, ypred, average='macro')
        f1 = f1_score(ytrue, ypred, average='macro')
        accuracy = accuracy_score(ytrue, ypred)
        if(self.model_args.is_test):
            save_predictions(ypred, ytrue, self.model_args.path_to_output_csv)
        return {'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy}

    def compute_metrics_multilabel(self, eval_pred, threshold = 0.5):

        logits, labels = eval_pred
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        logit_labels = np.zeros(probs.shape)
        logit_labels[np.where(probs >= threshold)] = 1
        if(self.model_args.is_test):
            ypred = [[i for i in range(len(item)) if item[i]==1] for item in logit_labels]
            ytrue = [[i for i in range(len(item)) if item[i]==1] for item in labels]
            save_predictions(ypred, ytrue, self.model_args.path_to_output_csv)
        return evaluate_multilabel_classifier(logit_labels, labels)

    def compute_metrics(self, eval_pred):
        if(self.model_args.problem_type == "multi_label_classification"):
            return self.compute_metrics_multilabel(eval_pred)
        else:
            return self.compute_metrics_multiclass(eval_pred)

    def finetune_model(self):

        print("Preprocessing dataset!")
        
        train_dataset, eval_dataset = self.dataset['train'], self.dataset['validation']
        if self.model_args.synthetic_usage == 'synthetic-train-augment':
            train_dataset = concatenate_datasets([self.dataset['train'], self.dataset['synthetic']])
        elif self.model_args.synthetic_usage == 'synthetic-train-only':
            train_dataset = self.dataset['synthetic']
        processed_train_dataset = self.process_map(train_dataset)
        processed_eval_dataset = self.process_map(eval_dataset)
        self.model = self.model.to(self.device)

        print("Model training begins!")

        trainer = Trainer(model = self.model, args = self.training_args,
                          train_dataset = processed_train_dataset,
                          eval_dataset = processed_eval_dataset,
                          compute_metrics = self.compute_metrics,)
               
        #trainer.train(resume_from_checkpoint = True)
        trainer.train()
        trainer.save_model(self.model_args.path_to_model)
    
    def test_model(self):

        print("Preprocessing dataset")

        processed_test_dataset = self.process_map(self.dataset)
        self.model = self.model.to(self.device)

        print("Model evaluation begins!")

        trainer = Trainer(model=self.model,
                        args=self.training_args,
                        compute_metrics=self.compute_metrics,
                        eval_dataset=processed_test_dataset,)
        
        evaluation_results = trainer.evaluate()
        print("Evaluation results:", evaluation_results)
        evaluation_results['model_name'] = self.model_args.path_to_model
        df = pd.DataFrame([evaluation_results])
        if(os.path.exists(self.model_args.path_to_aggregated_results)):
            df.to_csv(self.model_args.path_to_aggregated_results, index=False, header = None, mode = 'a')
        else:
            df.to_csv(self.model_args.path_to_aggregated_results, index=False, mode = 'a')

if __name__ == "__main__":
        arg_parser = transformers.HfArgumentParser((TrainingArguments, MiscArguments))

        train_args, model_args = arg_parser.parse_args_into_dataclasses()
        args = Arguments(train=train_args, model=model_args)
        print("Initialization...")
        if(args.model.is_train):
          print("Training:\n")
          obj = ModelFT(args = args)
          obj.finetune_model()
        if(args.model.is_test):
          print("Testing:\n")
          obj = ModelFT(args = args)
          obj.test_model()