from synthtexteval.eval.text_quality.arguments import MauveArgs
import mauve
import pickle
import pandas as pd

#TODO: Save the arguments somewhere as well
#TODO: Copy entire list of Mauve arguments

def calculate_mauve_score(df,  args: MauveArgs):

    """
    Estimates Mauve score for a given list of source and reference texts

    Arguments:
        df (pd.DataFrame): A pandas DataFrame containing source and reference texts.
        args (MauveArgs) : An object defining the MAUVE hyperparameters for the setup.

    Returns:
        pd.DataFrame: A DataFrame containing the MAUVE score for each source-reference pair.
    """
    source_texts, reference_texts = df[args.source_text_column].tolist(), df[args.ref_text_column].tolist()
    
    out = mauve.compute_mauve(p_text=source_texts, q_text=reference_texts, device_id=args.device_id, featurize_model_name = args.model_name_featurizer, max_text_length=args.max_text_length, verbose=False)

    return (out, out.mauve)