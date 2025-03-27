from synthtexteval.eval.text_quality.mauve_metric import calculate_mauve_score
from synthtexteval.eval.text_quality.perplexity import calculate_perplexity
from synthtexteval.eval.text_quality.frechet import calculate_fid_score
from synthtexteval.eval.text_quality.arguments import MauveArgs, LMArgs, FrechetArgs
import pandas as pd
import pickle

def test():
    test_cases = {
        'mauve': (calculate_mauve_score, MauveArgs),
        'perplexity': (calculate_perplexity, LMArgs),
        'frechet': (calculate_fid_score, FrechetArgs)
    }
    
    # Loop through the dictionary to test each function
    for test_name, (test_func, args_class) in test_cases.items():
        print(f"Testing {test_name}...")
        
        # Create an instance of the argument dataclass
        test_args = args_class()

        test_df = pd.DataFrame({'source': ['The dog ran after the cat.', 'The Eiffel Tower is one of the tallest buildings in the world.'],
                       'reference': ['The dog chased the cat.', 'The Eiffel Tower is in Paris used to be one of the tallest buildings in the world.']})

        # Call the function with the dataframe and arguments
        result = test_func(test_df, test_args)
        
        # Print the result for testing purposes
        print(result)

        # Save the result to a pickle file
        with open(test_args.output_pkl_file_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Result saved to {test_args.output_pkl_file_path}\n")

test()