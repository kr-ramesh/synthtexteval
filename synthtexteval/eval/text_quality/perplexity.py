from evaluate import load
from synthtexteval.eval.text_quality.arguments import LMArgs
import pandas as pd
import pickle

def calculate_perplexity(df, args: LMArgs):
    """
    Calculates the perplexity for the outputs from a language model.

    Arguments:
        df (pd.DataFrame): A pandas DataFrame containing the source texts
        args (LMArgs) : An object defining the LM-based metrics' hyperparameters for the setup.

    Returns:
        results (dict) : Results from the LM metrics' score for each source-reference pair.
    """
    
    texts = df[args.source_text_column].tolist()
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=texts, model_id=args.model_name)
    
    if(args.ref_text_column):
        perplexity = load("perplexity", module_type="metric")
        res = perplexity.compute(predictions=df[args.ref_text_column].tolist(), model_id=args.model_name)
        return (results, results['mean_perplexity'], res, res['mean_perplexity'])

    return (results, results['mean_perplexity'])

