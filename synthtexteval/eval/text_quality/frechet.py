from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from synthtexteval.eval.text_quality.arguments import FrechetArgs
from typing import List, Tuple
import torch
import pickle
import pandas as pd
import numpy as np

def frechet_inception_distance_score(real_embeddings: List[torch.Tensor], 
                  synth_embeddings: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
    
    """
    Computes the Frechet Inception Distance given a list of real and
    synthetic embedding representations of the source and reference documents.
    """
    real_avg_embedding = sum(real_embeddings)/len(real_embeddings)
    real_scores = [util.pytorch_cos_sim(real_avg_embedding, real_embedding) for real_embedding in real_embeddings]
    synth_scores = [util.pytorch_cos_sim(real_avg_embedding, synth_embedding) for synth_embedding in synth_embeddings]
        
    return {'real_scores': [i.item() for i in real_scores], 'synth_scores': [i.item() for i in synth_scores]}

def calculate_fid_score(df, args: FrechetArgs):
    
    """
    Estimates the Frechet Inception Distance scores for a given list of source and reference texts

    Arguments:
        df (pd.DataFrame): A pandas DataFrame containing source and reference texts.
        args (FrechetArgs) : An object defining the Frechet hyperparameters for the setup.

    Returns:
        pd.DataFrame: A DataFrame containing the Frechet score for each source-reference pair.
    """

    synthetic_texts, real_texts = df[args.source_text_column].tolist(), df[args.ref_text_column].tolist()
    
    sent_model = SentenceTransformer(args.sent_transformer_model_name)
    synth_text_embedding = sent_model.encode(synthetic_texts)
    real_text_embedding = sent_model.encode(real_texts)
    
    fid_score_result = frechet_inception_distance_score(synth_text_embedding, real_text_embedding)

    return (fid_score_result, np.mean(fid_score_result['synth_scores']))
