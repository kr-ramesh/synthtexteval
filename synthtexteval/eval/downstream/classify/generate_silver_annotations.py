import os, torch
import pandas as pd
import numpy as np
from datasets import Dataset
from synthtexteval.eval.downstream.classify.utils import load_model, read_data


def annotate(model, data, problem_type, threshold=0.5):
    """
    Annotates the dataframe with the model predictions
    """
    # Generate predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**data)
        if problem_type == 'multi_label_classification':
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(outputs.logits))
            labels = np.zeros(probs.shape)
            labels[np.where(probs >= threshold)] = 1
            # Convert label from one-hot to a list of labels
            labels = [list(np.where(label == 1)[0]) for label in labels]
        else:
            predictions = np.argmax(outputs.logits, axis=-1)
            labels = predictions.tolist()  

    return labels

def save_annotated_data(df, output_path):
    """Saves the annotated dataframe to a CSV file
    Args:
        - df (pd.DataFrame): The annotated dataframe
        - output_path (str): The path to save the annotated dataframe
    """
    df.to_csv(output_path, index=False)
    print(f"Annotated data saved to {output_path}")

def generate_silver_annotations(model_name, path_to_model, n_labels, problem_type, data_path, text_column, label_column, output_path, ckpt_exists = True):
    
    """Generates silver annotations for a given dataset using a HuggingFace model
    Args:
        - model_name (str): The name of the HuggingFace model to be used for annotation
        - path_to_model (str): The path to the model to be loaded, in case a checkpoint is provided
        - n_labels (int): The number of labels for the classification task
        - problem_type (str): The type of classification task (e.g., 'single_label_classification')
        - data_path (str): The path to the dataset to be annotated
        - text_column (str): The name of the column containing the text to be classified
        - label_column (str): The name of the column to store the predicted labels
        - output_path (str): The path to save the annotated dataframe
        - ckpt_exists (bool): Whether a checkpoint exists for the model
    """
    # Load the model and dataset
    model, tokenizer = load_model(model_name, path_to_model, n_labels, problem_type, ckpt_exists)
    print("Model loaded successfully.")

    # Read the data
    data = read_data(data_path)
    #data = data['test'].select(range(10))

    # Tokenize the data
    dataset = tokenizer(data[text_column], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    print("Data tokenized successfully.")

    # Annotate the dataset
    annotations = annotate(model, dataset, problem_type)

    df = pd.read_csv(data_path)
    df['Annotated Label'] = annotations
    print("Dataset annotated successfully.")

    # Save the annotated dataset
    save_annotated_data(df, output_path)
    print("Silver annotations generated successfully.")