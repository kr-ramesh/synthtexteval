from load_utils import read_data, load_model, tokenize_data, evaluate_multilabel_classifier
from typing import Any, Callable, List, Optional, Union, Dict, Sequence
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import transformers, evaluate
import argparse, torch
import numpy as np
import pandas as pd
import os

os.environ["WANDB_DISABLED"] = "true"

#TODO: Comment the code
#TODO: Shift around function classes, make the code look neater

@dataclass
class MiscArguments:
    model_name: str = field(default="bert-base-uncased", metadata={
        "help": "Model name in HuggingFace"
    })
    dataset_name: str = field(default="mimic", metadata={
        "help": "Name of the dataset"
    })
    path_to_dataset: str = field(default="N/A", metadata={
        "help": "Path to dataset directory"
    })
    path_to_model: str = field(default="N/A", metadata={
        "help": "Path to HuggingFace model to be trained"
    })
    csv_output_path: str = field(default="outputs.csv", metadata={
        "help": "Path to where the downstream outputs are saved"
    })
    n_labels: int = field(default=10, metadata={
        "help": "Number of classes"
    })
    is_train: Optional[bool] = field(default = False, metadata={
        "help": "If set, the model is fully fine-tuned over the dataset specified in the train.csv in the path_to_dataset directory."
    })
    is_test: Optional[bool] = field(default = False, metadata={
        "help": "If set, the model is tested over the dataset specified in the test.csv in the path_to_dataset directory."
    })

class ModelFT():

    def __init__(self, model_name, n_labels, dataset_name, path_to_dataset, training_args, path_to_model = "N/A", is_test = False, csv_output_path = "inf-csvs/outputs.csv"):

        self.dataset_name = dataset_name
        self.path_to_dataset = path_to_dataset
        self.is_test = is_test
        self.n_labels = n_labels
        
        if(self.is_test):
            self.dataset, self.n_labels = read_data(data_dir = path_to_dataset, n_labels = self.n_labels, dataset_name = dataset_name, is_test = True)
            self.dataset = self.dataset['test']
            self.model, self.tokenizer = load_model(model_name = model_name, path_to_model = path_to_model, n_labels = self.n_labels, problem_type = dataset_name, ckpt_exists = True)
        else:
            self.dataset, n_labels = read_data(data_dir = self.path_to_dataset, n_labels = self.n_labels, dataset_name = self.dataset_name, is_test = False)
            self.model, self.tokenizer = load_model(model_name = model_name, path_to_model = path_to_model, n_labels = n_labels, problem_type = dataset_name)
        
        #self.batch_size = 8 #TODO: Where is this argument being used?
        #self.max_length = 512

        self.device = "cuda"
        self.training_args = training_args
        self.metric = evaluate.load("accuracy")
        self.path_to_model = path_to_model
        self.csv_output_path = csv_output_path
        if(self.dataset_name == "mimic"):
            self.text_field, self.label_field= "TEXT", "Label"

    def process_map(self, data):
        processed_datasets = data.map(self.preprocess_function,
                                              batched=True,
                                              num_proc=1,
                                              load_from_cache_file=False,
                                              desc="Running tokenizer on dataset",)

        processed_datasets = processed_datasets.remove_columns(data.column_names)

        return processed_datasets

    def preprocess_function(self, examples):

        train_input_ids, train_class_labels, train_attention_masks = tokenize_data(self.tokenizer, examples[self.text_field], examples[self.label_field], self.dataset_name)
        examples['input_ids'] = train_input_ids
        examples['labels'] = train_class_labels
        examples['attention_mask'] = train_attention_masks

        return examples

    def compute_metrics_multiclass(self, eval_pred):

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions = predictions, references = labels)

    def compute_metrics_multilabel(self, eval_pred, threshold = 0.5):

        logits, labels = eval_pred
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        logit_labels = np.zeros(probs.shape)
        logit_labels[np.where(probs >= threshold)] = 1
        if(self.is_test):
            print("Saving file!")
            ypred = [[i for i in range(len(item)) if item[i]==1] for item in logit_labels]
            ytrue = [[i for i in range(len(item)) if item[i]==1] for item in labels]
            df = pd.DataFrame({'Predicted': ypred, 'Ground': ytrue})
            df.to_csv(self.csv_output_path)
        return evaluate_multilabel_classifier(logit_labels, labels)
        #save_results_to_file(res, self.csv_output_path)

    def compute_metrics(self, eval_pred):
        
        #Specified different metrics for different datasets, based on the type of task.
        if(self.dataset_name == "mimic"):
            return self.compute_metrics_multilabel(eval_pred)
        elif(self.dataset_name == "cps"):
            return self.compute_metrics_multiclass(eval_pred)

    def finetune_model(self):

        print("Preprocessing dataset!")
        # eval on subset of classes for which we have perturbed samples
        train_dataset, eval_dataset = self.dataset['train'], self.dataset['validation']

        processed_train_dataset = self.process_map(train_dataset)
        processed_eval_dataset = self.process_map(eval_dataset)
        self.model = self.model.to(self.device)

        print("Model training begins!")

        trainer = Trainer(model = self.model, args = self.training_args,
                          train_dataset = processed_train_dataset,
                          eval_dataset = processed_eval_dataset,
                          compute_metrics = self.compute_metrics,)

        #TODO: Can enable option to resume training from a previous checkpoint. Not req'd in current version.    
        #trainer.train(resume_from_checkpoint = True)
        trainer.train()
        trainer.save_model(self.path_to_model)
    
    def test_model(self):

        print("Preprocessing dataset")

        processed_test_dataset = self.process_map(self.dataset)
        self.model = self.model.to(self.device)

        print("Model evaluation begins!")

        trainer = Trainer(model=self.model,
                        args=self.training_args,
                        compute_metrics= self.compute_metrics,
                        eval_dataset=processed_test_dataset,)
        
        evaluation_results = trainer.evaluate()
        print("Evaluation results:", evaluation_results)
        evaluation_results['model_name'] = self.path_to_model
        df = pd.DataFrame([evaluation_results])
        
        if(os.path.exists(self.csv_output_path)):
            df.to_csv(self.csv_output_path, index=False, header = None, mode = 'a')
        else:
            df.to_csv(self.csv_output_path, index=False, mode = 'a')

if __name__ == "__main__":
        arg_parser = transformers.HfArgumentParser((MiscArguments))

        train_args, args = TrainingArguments(output_dir="top3", evaluation_strategy="epoch", num_train_epochs = 3, metric_for_best_model = "f1", per_device_eval_batch_size=8), arg_parser.parse_args_into_dataclasses()[0]
        train_args = train_args.set_save(strategy="steps", total_limit = 2)
        print("Initialization...")
        if(args.is_train):
          obj = ModelFT(model_name = args.model_name, n_labels = args.n_labels, dataset_name = args.dataset_name, path_to_dataset = args.path_to_dataset, training_args = train_args, path_to_model = args.path_to_model, is_test = False)
          obj.finetune_model()
        if(args.is_test):
          print("Is testing!")
          obj = ModelFT(model_name = args.model_name, n_labels = args.n_labels, dataset_name = args.dataset_name, path_to_dataset = args.path_to_dataset, training_args = train_args, path_to_model = args.path_to_model, is_test = True, csv_output_path=args.csv_output_path)
          obj.test_model()
          #test_model_multilabel(training_args = train_args, model_name = args.model_name, path_to_model = args.saved_path_to_model, path_to_dataset = args.path_to_dataset, dataset_name = args.dataset_name, device = "cuda", csv_output_path = args.csv_output_path)
