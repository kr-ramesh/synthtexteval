import synthtexteval
import pandas as pd
import numpy as np
from synthtexteval.eval.fairness.metrics import analyze_group_fairness_performance

def generate_single_label_test_df(n=100, c = 2):
    """
    Generates a single label test DataFrame with 'n' samples and 'c' classes.
    """
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        'Ground': np.random.randint(0, c, size=n),  # True class labels
        'Predicted': np.random.randint(0, c, size=n)  # Model predictions
    })
    return df

def generate_multilabel_test_df(n=100, c=3):
    """
    Generates a multilabel test DataFrame with 'n' samples and 'c' classes.
    Each sample has multiple labels represented as lists.
    """
    np.random.seed(42)  # For reproducibility
    
    def generate_labels():
        num_labels = np.random.randint(1, c + 1)  # Random number of labels per sample
        return list(np.random.choice(range(0, c), size=num_labels, replace=False))
    
    ground_truth = [generate_labels() for _ in range(n)]
    predictions = [generate_labels() for _ in range(n)]
    
    df = pd.DataFrame({
        'Ground': ground_truth,
        'Predicted': predictions
    })
    
    return df

# Example: Generate a dataset with 100 points and 3 classes
df_single = generate_single_label_test_df(n=100, c=2)
df_single['Subgroup'] = np.random.choice(['a', 'b', 'c'], size=len(df_single))
p_df, f_df = analyze_group_fairness_performance(df_single, problem_type = "single_label", num_classes = 2)
print(p_df.head())
print(f_df.head())


# Example usage:
df_multilabel = generate_multilabel_test_df(n=100, c=3)
df_multilabel['Subgroup'] = np.random.choice(['a', 'b', 'c'], size=len(df_multilabel))
p_df, f_df = analyze_group_fairness_performance(df_multilabel, problem_type = "multilabel", num_classes = 3)
print(p_df.head())
print(f_df.head())
