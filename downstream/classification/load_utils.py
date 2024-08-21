from datasets import load_dataset, load_from_disk, Features, Value, ClassLabel, Dataset, DatasetDict
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch, ast
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score
import collections, operator, statistics


def read_data(data_dir, n_labels, dataset_name = "mimic", is_test = True):

      """Function to load and return the dataset for training or the evaluation of bias

       Args:
            - dataset_name (str): Name of the dataset to be loaded
            - partition_type (str) : Partition of dataset to return (set to 'testing purposes' in order to test your scripts. Set to 'all' to return every partition.)
       Returns:
            - dataset (lst): List of multiple json objects contained in the pickle file

      """
      
      if(dataset_name == "mimic"):
          if(is_test):
            print("Loading test data")
            dataset = load_dataset('csv', data_files = {"test": data_dir + "test.csv"})
          else:
            dataset = load_dataset('csv', data_files = {"train": data_dir + "train.csv", "validation" : data_dir + 'eval.csv'})

          #dataset = dataset.remove_columns(['Unnamed: 0', 'Unnamed: 0.1', 'DIAGNOSIS', 'LONG_TITLE']) 
          lst = n_labels
          print("Number of labels", n_labels)
          return dataset, lst


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

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

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


def test_model(model_name = "bert-base-uncased", path_to_model ="N/A", path_to_dataset = "", dataset_name = "mimic", device = "cuda", batch_size = 8, csv_output_path = "outputs.csv"):

      """Testing the model over the evaluation set

       Args:
            - model_name (str): Name of the model to be loaded
            - path_to_model (str): Path to the model to be loaded, in case a checkpoint is provided

      """
      dataset, n_labels = read_data(data_dir = path_to_dataset, dataset_name = dataset_name)
      dataset = dataset['test']
      
      model, tokenizer = load_model(model_name = model_name, path_to_model = path_to_model, n_labels = n_labels, ckpt_exists = True) #accept labels from the user or automatically calculate it
      if torch.cuda.is_available():
        model = model.to(device)

      print("Batching data:")
      test_input_ids, test_class_labels, test_attention_masks = tokenize_data(tokenizer, dataset['TEXT'], dataset['Label']) #TODO: change the fields
      test_dataset = TensorDataset(test_input_ids, test_class_labels, test_attention_masks)
      test_dataloader = DataLoader(test_dataset, batch_size = batch_size)

      print("Evaluation begins!")

      model.eval()
      # Tracking variables
      total_eval_accuracy, total_eval_loss = 0, 0
      y_test_prob, actual_label_list, pred_label_list = [], [], []


      # Evaluate data for one epoch
      for batch in test_dataloader:
            if torch.cuda.is_available():
              b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[2].to(device), batch[1].to(device)
            else:
              b_input_ids, b_input_mask, b_labels = batch[0], batch[2], batch[1]
            with torch.no_grad():

              outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,)

            total_eval_loss += outputs.loss.item()
            logits = outputs.logits.detach().cpu().numpy()

            probs = softmax(logits, axis=1)
            pred_flat = list(np.argmax(logits, axis=1).flatten())
            label_ids = list(b_labels.to('cpu').numpy())
            label_ids2 = b_labels.to('cpu').numpy()

            for ind, lab in enumerate(label_ids):
              y_test_prob.append(probs[ind][lab])


            actual_label_list = (actual_label_list + label_ids)
            pred_label_list = (pred_label_list + pred_flat)
            total_eval_accuracy += flat_accuracy(logits, label_ids2)

      avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
      print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

      print("Saving file!")
      df = pd.DataFrame({'Predicted': pred_label_list, 'Ground': actual_label_list})
      df.to_csv(csv_output_path)
      

