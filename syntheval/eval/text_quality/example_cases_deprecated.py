"""from syntheval.eval.text_quality.mauve_metric import calculate_mauve_score
from syntheval.eval.text_quality.perplexity import calculate_perplexity
from syntheval.eval.text_quality.frechet import calculate_fid_score
from syntheval.eval.text_quality.arguments import MauveArgs, LMArgs, FrechetArgs
import pandas as pd
import pickle

test_df = pd.DataFrame({'source': ['The dog ran after the cat.', 'The Eiffel Tower is one of the tallest buildings in the world.'],
                       'reference': ['The dog chased the cat.', 'The Eiffel Tower is in Paris used to be one of the tallest buildings in the world.']})

result = calculate_mauve_score(test_df, MauveArgs)

test_df = pd.DataFrame({'source': ['The dog ran after the cat.', 'The Eiffel Tower is one of the tallest buildings in the world.'],
                       'reference': ['The dog chased the cat.', 'The Eiffel Tower is in Paris used to be one of the tallest buildings in the world.']})

result = calculate_fid_score(test_df, FrechetArgs)

test_df = pd.DataFrame({'source': ['The dog ran after the cat.', 'The Eiffel Tower is one of the tallest buildings in the world.'],
                       'reference': ['The dog chased the cat.', 'The Eiffel Tower is in Paris used to be one of the tallest buildings in the world.']})

result = calculate_perplexity(test_df, LMArgs)"""