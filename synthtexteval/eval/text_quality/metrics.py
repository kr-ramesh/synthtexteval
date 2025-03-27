from synthtexteval.eval.text_quality.mauve_metric import calculate_mauve_score
from synthtexteval.eval.text_quality.perplexity import calculate_perplexity
from synthtexteval.eval.text_quality.frechet import calculate_fid_score
from tabulate import tabulate
import pandas as pd

metrics = ['mauve', 'fid', 'synth perplexity', 'real perplexity']

class TextQualityEval():

    def __init__(self, args):
        self.args = args
        self.results = {i: [] for i in metrics}
        self.detailed_results = {i: [] for i in metrics}
    
    def calculate_mauve_score(self, df):
        result = calculate_mauve_score(df, self.args.mauve)
        self.results['mauve'].append(result[-1])
        self.detailed_results['mauve'].append(result[0])
    
    def calculate_fid_score(self, df):
        result = calculate_fid_score(df, self.args.frechet)
        self.results['fid'].append(result[-1])
        self.detailed_results['fid'].append(result[0])
    
    def calculate_perplexity(self, df):
        result = calculate_perplexity(df, self.args.perplexity)
        self.results['synth perplexity'].append(result[1])
        if(len(result)>2):
            self.results['real perplexity'].append(result[-1])
            self.detailed_results['real perplexity'].append(result[-2])
        #self.results with .3f
        self.detailed_results['synth perplexity'].append(result[0])
    
    def print_metrics(self, dict_df):
        dict_df = {k: v for k, v in self.results.items() if v}  # Filter out empty lists
        dict_df = {k: v if isinstance(v[0], str) else [f"{i:.3f}" for i in v] for k, v in dict_df.items()}
        print("Automated Open-Ended Text Evaluation Metrics:")
        print(tabulate(dict_df, headers="keys", tablefmt="pretty"))
    
    def return_results(self):
        return self.results
    


    