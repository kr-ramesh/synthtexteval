from syntheval.qual.mauve_metric import calculate_mauve_score
from syntheval.qual.perplexity import calculate_perplexity
from syntheval.qual.frechet import calculate_fid_score

class QualEval():

    def __init__(self, args):
        self.args = args
        self.results = {}
    
    def calculate_mauve_score(self, df):
        self.results['mauve'] = calculate_mauve_score(df, self.args.MauveArgs)
    
    def calculate_fid_score(self, df):
        self.results['fid'] = calculate_fid_score(df, self.args.FrechetArgs)
    
    def calculate_perplexity(self, df):
        self.results['perplexity'] = calculate_perplexity(df, self.args.LMArgs)
    


    