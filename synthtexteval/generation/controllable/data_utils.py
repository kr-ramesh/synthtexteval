
import datasets
import evaluate
import torch
import numpy as np
import pandas as pd
import random
import os, re, ast, json
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from synthtexteval.utils.utils import split_and_save_dataframe
from collections import Counter
from tqdm import tqdm
#TODO: Format the control code part of this

# Modified from https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
def main_preprocess_function(examples, tokenizer, text_field, prompt_begin, prompt_end, label_field, sequence_len, single_token=True):
    """
    Preprocess function for the main task of prompt tuning.
    This function will prepare the input for the model.
    """
    batch_size = len(examples[text_field])

    # Prepare the context with the text in between of prompts, e.g. "Sentence : <text> Label :"
    inputs = [prompt_begin + str(x) + prompt_end for x in examples[text_field]]

    # Prepare the prediction part
    targets = [str(x) for x in examples[label_field]]

    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    # Concatenate the context and prediction parts as one input and set -100 to the labels of the context part
    # This is because only the label part will be used to calculate the loss
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        if single_token:
            # Tokenizer adds <s> to input_ids so just take the last id
            # NOTE THAT THIS ASSUMES THE LABEL IS SINGLE TOKEN
            label_input_ids = [labels["input_ids"][i][-1]]
        else:
            # Tokenizer adds <s> to input_ids so just take the rest
            label_input_ids = labels["input_ids"][i][1:]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    # Pad the samples with sequence_len and trim if longer than sequence_len
    # NOTE THAT IF CONTEXT IS LONGER THAN SEQUENCE_LEN, THERE WILL BE NOTHING TO PREDICT, LABEL IS ALL -100
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            sequence_len - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (sequence_len - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (sequence_len - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:sequence_len])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:sequence_len])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:sequence_len])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class CustomDataset:
    dataset = None
    classes = None # List of class labels
    text_field = None # Name of the field in the dataset that contains the text
    prompt_begin = None # Prompt to add to the beginning of the text, e.g. "Sentence : "
    prompt_end = None # Prompt to add to the end of the text, e.g. " Label :"
    label_field = None # Name of the field in the dataset that contains the label
    evaluate = None # Evaluation metric

    def __init__(self, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def target_max_len(self):
        target_lens = [len(self.tokenizer(class_label)["input_ids"]) for class_label in self.classes]
        target_max_len = max(target_lens)
        return target_max_len

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may lead to a memory issue.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids
    
    def preprocess_function(self, example):
        return main_preprocess_function(example, self.tokenizer, self.text_field, self.prompt_begin,
                                         self.prompt_end, self.label_field, self.sequence_len, single_token=False)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Only keep predictions up to last token
        predictions = predictions[..., :-1]
        # Only keep labels from the first token
        labels = labels[..., 1:]
        # Replace -100 of the labels as we don't want the content
        predictions = np.where(labels != -100, predictions, self.tokenizer.pad_token_id)
        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute ROUGE scores
        result = self.evaluate.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return {k: round(v, 4) for k, v in result.items()}

    def compute_test_metrics(self, trainer, args):
        
        print(f"Testing for the entire dataset. Number of generations per prompt: {args.num_return_seq}")
        
        try:
            test_dataset = self.dataset['test']
        except:
            df = pd.read_csv(self.path_to_test_dataset)
            df = df[df[self.text_field].notna()]
            test_dataset = Dataset.from_pandas(df)
        if(trainer.args.dry_test_run):
                test_dataset = test_dataset.select(range(5))
                #num_return_seq = 2

        # Need to use this only if your text_field is 'label'
        if(self.text_field == "label"):
            test_dataset = test_dataset.map(lambda x: {self.text_field: test_dataset.features[self.text_field].int2str(x[self.text_field])})  
            new_features = test_dataset.features.copy()
            new_features[self.text_field] = datasets.Value("string")
            test_dataset = test_dataset.cast(new_features)
        
        # Ensuring that we retain attributes from other columns. Can add functionality later to accept this as an argument from the user.
        retain_columns = [col for col in test_dataset.column_names if col not in [self.label_field]]
        output_dataframe = {}
        for col in retain_columns:
            output_dataframe[col] = [element for element in test_dataset[col] for i in range(args.num_return_seq)]
        
        print("Length of test data", len(test_dataset))

        test_dataset = test_dataset.map(
            lambda x: {self.text_field: [self.prompt_begin + str(article) + self.prompt_end for article in x[self.text_field]]},
            batched=True,
            num_proc=None,
        )
        
        # Tokenize data
        def test_preprocess_function(examples):
            model_inputs = trainer.tokenizer(examples[self.text_field], padding=False)

            # 2. reserve the original article and summary for saving
            model_inputs[self.label_field] = examples[self.label_field]
            return model_inputs

        with trainer.args.main_process_first(desc="tokenizing test dataset"):
            test_dataset = test_dataset.map(
                test_preprocess_function,
                batched=True, num_proc=None, desc="tokenizing dataset",
                remove_columns=test_dataset.column_names)

        # Filter out samples too long, e.g. more than 750 tokens
        test_dataset = test_dataset.filter(lambda x: len(x['input_ids']) < 750)

        test_dataset.set_format(type="torch")

        def generate_batched(
            model,
            tokenizer,
            device,
            query_tensors,
            batch_size: int = 4,
            return_prompt: bool = True,
            pad_to_multiple_of: int = None,
            **generation_kwargs,
        ):
            outputs = []

            tokenizer.padding_side = "left"

            # handle distributed case and distribute query_tensors among gpus
            query_tensors = query_tensors[device.index::trainer.args.world_size]

            # in case we have fewer examples than bs
            batch_size = min(len(query_tensors), batch_size)

            for i in range(0, len(query_tensors), batch_size):
                # prevent overflow if query tensors are not even multiple of bs
                end_index = min(len(query_tensors), i + batch_size)

                batch = query_tensors[i:end_index]
                batch_mask = [torch.ones_like(element) for element in batch]
                inputs = {"input_ids": batch, "attention_mask": batch_mask}

                padded_inputs = tokenizer.pad(
                    inputs,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    #generations = model.generate(**padded_inputs, **generation_kwargs)
                    generations = model.generate(**padded_inputs, do_sample=True, min_new_tokens = args.min_new_tokens, max_new_tokens = args.max_new_tokens, top_k=50, top_p=0.95, num_return_sequences=args.num_return_seq)
                
                ind = 0
                for mask in padded_inputs["attention_mask"]:
                    for ind_item in range(ind, ind+args.num_return_seq):
                        output = generations[ind_item][(1 - mask).sum() :]  # remove padding

                        if not return_prompt:
                            output = output[(mask).sum() :]  # remove prompt
                        outputs.append(output)
                    ind+=args.num_return_seq
            return outputs

        if hasattr(trainer.model, "generate"):
            model = trainer.model
        # The following is for GradSampleModule wrapping
        elif hasattr(trainer.model._module, "generate"):
            model = trainer.model._module
        # The following is for GradSampleModule and DPDDP wrapping
        elif hasattr(trainer.model._module.module, "generate"):
            model = trainer.model._module.module
        else:
            raise ValueError("Cannot find generate function in the model.")

        model.eval()
        generation_kwargs = {"max_new_tokens": 100, "pad_token_id": trainer.tokenizer.pad_token_id,
                             "eos_token_id": trainer.tokenizer.eos_token_id,}

        response_tensors = generate_batched(
            model, trainer.tokenizer, trainer.args.device,
            test_dataset["input_ids"],
            batch_size=trainer.args.eval_batch_size, return_prompt=False,
            **generation_kwargs
        )
        responses = [trainer.tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                                    for r in response_tensors]
        input_data = [trainer.tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                      for r in test_dataset["input_ids"] for rep in range(args.num_return_seq)] #TODO: Return num_sequences in place of 3 in the range
        output_dataframe['input_prompt'], output_dataframe['output_text'] = input_data, responses
        #df = pd.DataFrame({'Input Prompt': input_data, 'Output Text': responses})
        df = pd.DataFrame(output_dataframe)
        return df

class HFDataset(CustomDataset):
    """
    Dataset class for HuggingFace datasets.
    """
    def __init__(self, args, tokenizer):

        self.dataset = load_dataset(args.data.path_to_dataset)
        self.control_field = args.data.control_field
        self.text_field = args.data.text_field
        self.prompt_begin = args.data.prompt_begin
        self.prompt_end = args.data.prompt_end
        self.label_field = args.data.label_field
        self.evaluate = evaluate.load("rouge")
                
        super().__init__(tokenizer, args.model.sequence_len)


class TAB(CustomDataset):
    """
    Dataset class for HuggingFace datasets.
    """
    def __init__(self, args, tokenizer):

        self.dataset = load_from_disk(args.data.path_to_dataset)
        self.control_field = "control"
        self.text_field = self.control_field
        self.prompt_begin = ""
        self.prompt_end = "\n"
        self.label_field = "text"
        self.evaluate = evaluate.load("rouge")
                
        super().__init__(tokenizer, args.model.sequence_len)


class MIMIC(CustomDataset):

    def __init__(self, args, tokenizer, create_dataset = False):

        self.path_to_dataset = args.data.path_to_dataset
        self.control_field = "ICD9_CODE"
        self.text_field = 'LONG_TITLE'
        self.prompt_begin = "Diagnosis: "
        self.prompt_end = " Summary :"
        self.label_field = 'TEXT'
        self.evaluate = evaluate.load("rouge")

        try:
            self.dataset = load_from_disk(self.path_to_dataset)
            print("Successfully loaded dataset!")
        except:
            print("Dataset does not exist. Please create it using specify_control_codes() and try again.")
        
        super().__init__(tokenizer, args.model.sequence_len)

    
    def create_dataset(self, path_to_save_dataset, is_top_freq = 3, 
                              num_codes = 1000, train_eval_split = 0.95):
        
        os.makedirs(path_to_save_dataset, exist_ok=True)
        df, df2 = pd.read_csv(self.path_to_dataset), []
        
        df['ICD9_CODE'] = df['ICD9_CODE'].apply(lambda x: ast.literal_eval(x))
        df['ICD9_CODE'] = [[str(x).strip() for x in lst] for lst in df['ICD9_CODE'].tolist()]
        item_counts = Counter([item for sublist in df['ICD9_CODE'].tolist() for item in sublist])
        control_codes = [item for item, _ in item_counts.most_common(is_top_freq)]
        print("Control codes:", control_codes)
        
        #check
        df2 = df[df[self.control_field].apply(lambda x: any(item in control_codes for item in x))]
        df2['ICD9_CODE'] = [[label for label in sublist if label in control_codes] for sublist in df2['ICD9_CODE']]
          
        #print("Number of control codes:", len(control_codes))
        
        df2= df2.sample(frac=1.0, random_state=42)
        df = df2.sample(frac = 0.90)
        df_test = df2.drop(df.index)

        df_train = df.sample(frac = train_eval_split)
        df_eval = df.drop(df_train.index)

        self.dataset = DatasetDict()
        self.dataset['train'], self.dataset['validation'], self.dataset['test'] = Dataset.from_pandas(df_train), Dataset.from_pandas(df_eval), Dataset.from_pandas(df_test)

        self.dataset.save_to_disk(path_to_save_dataset)
        
        print(f"Length of the training, validation, test sets:{len(df_train)}, {len(df_eval)}, {len(df_test)}")
        
        return self.dataset

class WikiBio(CustomDataset):
    """
    Dataset class for WikiBio dataset.
    """
    def __init__(self, args, tokenizer):

        self.path_to_dataset = args.data.path_to_dataset
        #Hardcoded for custom datasets for convenience
        self.control_field = 'Name'
        self.text_field = self.control_field
        self.label_field = 'Text'
        self.prompt_begin = "Generate a biography about: "
        self.prompt_end = " Biography :"
        self.evaluate = evaluate.load("rouge")

        self.dataset = DatasetDict()
        if(args.model.inference == False):
            self.path_to_model = args.model.path_to_save_model
            self.dataset['train'], self.dataset['validation'], _ = self.specify_control_codes()
        else:
            self.path_to_test_dataset = args.data.path_to_test_dataset
        
        super().__init__(tokenizer, args.model.sequence_len)

    
    def specify_control_codes(self):

        try:
            output_dir = self.path_to_model.split("_DP")[0] + '_data/'
            path_to_train_dataset, path_to_eval_dataset, path_to_test_dataset = output_dir + 'train.csv', output_dir + 'validation.csv', output_dir + 'test.csv'
            os.mkdir(output_dir)
        except:
            print("Directory exists: ", path_to_train_dataset)
            if(os.path.isfile(path_to_test_dataset) and os.path.isfile(path_to_train_dataset) and os.path.isfile(path_to_eval_dataset)):
                return Dataset.from_pandas(pd.read_csv(path_to_train_dataset)),  Dataset.from_pandas(pd.read_csv(path_to_eval_dataset)),  Dataset.from_pandas(pd.read_csv(path_to_test_dataset))
                #return Dataset.from_pandas(path_to_train_dataset),  Dataset.from_pandas(path_to_eval_dataset),  Dataset.from_pandas(path_to_test_dataset)
                
        df_train, df_eval, df_test = split_and_save_dataframe(pd.read_csv(self.path_to_dataset), output_dir)
        
        print("Length of the training set:", len(df_train))
        print("Length of the validation set:", len(df_eval))
        print("Length of the test set :", len(df_test))

        return  Dataset.from_pandas(df_train),  Dataset.from_pandas(df_eval),  Dataset.from_pandas(df_test)

ALL_DATASETS = {"hfhub" : HFDataset, "wiki": WikiBio, "tab": TAB, "mimic": MIMIC}
