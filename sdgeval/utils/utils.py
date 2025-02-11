from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score

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
      