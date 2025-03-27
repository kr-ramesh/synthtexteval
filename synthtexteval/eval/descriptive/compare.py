import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from synthtexteval.eval.descriptive.distributional_difference import kl_divergence, jaccard_similarity, cosine_similarity_between_texts

def text_length_comparison(texts_1, texts_2):
    len_1, len_2 = [len(text.split()) for text in texts_1], [len(text.split()) for text in texts_2]
    return len_1, len_2

def basic_comparison_metrics(texts_1, texts_2, plot=False):
    """
    Compare basic metrics based on text length, etc., between two text distributions.
    Arguments:
        texts_1 (list): List of texts in the first distribution.
        texts_2 (list): List of texts in the second distribution.
        plot (bool): if true, save metric plot, else print out metric table
    Returns:
        pd.DataFrame: DataFrame containing comparison metrics.
    """
    len_1, len_2 = text_length_comparison(texts_1, texts_2)
    unique_words_1, unique_words_2 = [len(set(text.split())) for text in texts_1], [len(set(text.split())) for text in texts_2]
    
    # Compute the average, min, and max length
    avg_len_1, avg_len_2 = np.mean(len_1), np.mean(len_2)
    min_len_1, min_len_2 = np.min(len_1), np.min(len_2)
    max_len_1, max_len_2 = np.max(len_1), np.max(len_2)
    
    avg_unique_1, avg_unique_2 = np.mean(unique_words_1), np.mean(unique_words_2)
    
    # Generate comparison metrics in a plot
    if plot:
        fig, axs = plt.subplots(4, 1, figsize=(4,4))
        axs[0].barh(['Synthetic', 'Reference'], [avg_len_1, avg_len_2], color=['tab:green','tab:blue'])
        axs[1].barh(['Synthetic', 'Reference'], [min_len_1, min_len_2], color=['tab:green','tab:blue'])
        axs[2].barh(['Synthetic', 'Reference'], [max_len_1, max_len_2], color=['tab:green','tab:blue'])
        axs[3].barh(['Synthetic', 'Reference'], [avg_unique_1, avg_unique_2], color=['tab:green','tab:blue'])
        for i, metric in enumerate(["Avg. Length", "Min Length", "Max Length", "Avg. Unique Words"]):
            axs[i].set_xlabel(metric)
        plt.tight_layout()
        plt.savefig('./descriptive_metrics.png')
    
    # Generate comparison metrics in a dictionary
    else:
        comparison = {
            "Metric": ["Avg. Length", "Min Length", "Max Length", "Avg. Unique Words"],
            "Text Distribution 1": [avg_len_1, min_len_1, max_len_1, avg_unique_1],
            "Text Distribution 2": [avg_len_2, min_len_2, max_len_2, avg_unique_2],
            "Difference": [avg_len_1 - avg_len_2, min_len_1 - min_len_2, max_len_1 - max_len_2, avg_unique_1 - avg_unique_2]
        }
        
        # Convert to DataFrame for easy viewing
        comparison_df = pd.DataFrame(comparison)
        
        print(tabulate(comparison_df, headers="keys", tablefmt="grid", showindex=False))
    
        #return comparison_df

def compare_distributions(texts_1, texts_2, metrics):

    if('kl_divergence' in metrics):
        kl = kl_divergence(texts_1, texts_2)
        print(f"Kullback-Leibler Divergence: {kl:.3f}")
    if('jaccard' in metrics):
        js = jaccard_similarity(texts_1, texts_2)
        print(f"Jaccard similarity: {js:.3f}")
    if('cosine' in metrics):
        cs = cosine_similarity_between_texts(texts_1, texts_2)
        print(f"Cosine similarity: {cs:.3f}")