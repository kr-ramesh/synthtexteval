import sys
import pandas as pd
from synthtexteval.eval.text_quality.metrics import TextQualityEval
from synthtexteval.eval.text_quality.arguments import MauveArgs, LMArgs, FrechetArgs, Arguments
from dataclasses import dataclass
from datasets import load_from_disk

# Define the real and synthetic text data
df = pd.DataFrame({})
synthetic_samples = pd.read_csv(sys.argv[1])
real_samples = load_from_disk(sys.argv[1])

real_samples = real_samples.filter(lambda x: len(x['text']) < 3000)
len_samples = len(synthetic_samples) if len(synthetic_samples)<len(real_samples['train']) else len(real_samples['train'])
synthetic_samples = synthetic_samples.head(len_samples)
real_samples = real_samples['train'].select(range(len_samples))

df['source'] = synthetic_samples['output_text']
df['reference'] = real_samples['text']

args = Arguments(FrechetArgs, MauveArgs, LMArgs)
qual_estimator = TextQualityEval(args)
args.LMArgs.model_name = 'gpt2'

qual_estimator.calculate_perplexity(df)
qual_estimator.calculate_fid_score(df)
qual_estimator.calculate_mauve_score(df)

# Prints the results
qual_estimator.print_metrics(qual_estimator.return_results())