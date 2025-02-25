from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import json
import os

def create_classification_dataset(df, label_column, output_json_path, output_dir, 
                                  multilabel=False, separator=",", train_ratio=0.8, val_ratio=0.1, 
                                  test_ratio=0.1, random_state=42):
    """
    Creates a classification dataset from a DataFrame and saves the splits to a directory.
    """
                    
    df, _ = encode_labels(df, label_column = label_column, output_json_path = output_json_path, 
                          multilabel = multilabel, separator = separator)

    train_df, eval_df, test_df = split_and_save_dataframe(df, output_dir, train_ratio = train_ratio, 
                                      test_ratio = test_ratio, val_ratio=val_ratio)
    
    return train_df, eval_df, test_df

def encode_labels(df, label_column, output_json_path, multilabel=False, separator=","):
    """
    Encodes labels in a dataframe, creating a mapping of label to numeric value.

    Args:
        df (pd.DataFrame): Input dataframe with labels.
        label_column (str): Name of the column containing labels.
        output_json_path (str): Path to save the label mapping JSON.
        multilabel (bool): Whether it's a multilabel classification task.
        separator (str): Separator for multilabel values in a string format (default is ",").

    Returns:
        pd.DataFrame: Modified dataframe with numeric labels.
        dict: Mapping of labels to numeric values.
    """

    if multilabel:
        # Extract unique labels from all rows
        unique_labels = set()
        df[label_column].astype(str).apply(lambda x: unique_labels.update(x.split(separator)))
        unique_labels = sorted(unique_labels)  # Ensure consistent ordering
    else:
        unique_labels = sorted(df[label_column].unique())  # Unique labels for single-label case

    # Create label mapping
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Convert labels to numeric values
    if multilabel:
        df["Label"] = df[label_column].astype(str).apply(lambda x: [label_mapping[label] for label in x.split(separator)])
    else:
        df["Label"] = df[label_column].map(label_mapping)

    # Save label mapping as JSON
    #os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(label_mapping, f, indent=4)

    return df, label_mapping

# Example usage:
# df = pd.DataFrame({"Category": ["cat", "dog", "fish", "dog", "cat"]})  # Single label
# df, mapping = encode_labels(df, "Category", "label_mapping.json")

# df_multi = pd.DataFrame({"Categories": ["cat,dog", "dog,fish", "cat,fish", "dog", "cat"]})  # Multi-label
# df_multi, mapping_multi = encode_labels(df_multi, "Categories", "multi_label_mapping.json", multilabel=True)

def split_and_save_dataframe(df, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Splits a DataFrame into train, validation, and test sets and saves them to a directory.

    Args:
        df (pd.DataFrame): The input dataframe.
        output_dir (str): Directory to save the splits.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        random_state (int): Random seed for reproducibility.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Split into train and and (validation + test)
    train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=random_state, shuffle=True)
    test_size_relative = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=test_size_relative, random_state=random_state, shuffle=True)

    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Data saved to {output_dir}")
    print(f"Train: {len(train_df)} samples, Validation: {len(val_df)} samples, Test: {len(test_df)} samples")

    return train_df, val_df, test_df

def evaluate_multilabel_classifier(y_true, y_pred):
    """
    Evaluate a multilabel classifier using various metrics.
    """
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

def compare_models(model_1, model_2):
    """
    Compare two PyTorch models to see if they are the same. This was defined for debugging purposes.
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')