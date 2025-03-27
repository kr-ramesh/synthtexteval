import synthtexteval.generation.controllable.data_utils as data_utils
import synthtexteval.generation.controllable.argument_utils as argument_utils
import numpy as np
import pandas as pd
from structure_mimic import structured_format

np.random.seed(42)

args = argument_utils.Arguments(model = argument_utils.ModelArguments(), data = argument_utils.DataArguments(), train = None, privacy = None, lora = None)
args.data.dataset_name = "mimic"
args.data.path_to_dataset = "../../data/generator/data/mimic/MIMIC-NOTES.csv"

# STEP: Preprocessing dataset
"""df = pd.read_csv(args.data.path_to_dataset)

# Define aggregation dictionary to aggregate other columns with 'first' function
agg_dict = {col: 'first' for col in df.columns if col not in ['ICD9_CODE', 'LONG_TITLE', 'TEXT']}
# Group data by 'TEXT' column and aggregate other columns using the defined aggregation dictionary
grouped_data = df.groupby('TEXT').agg({**agg_dict,
                                         'ICD9_CODE': lambda x: list(x),
                                         'LONG_TITLE': lambda x: list(x)}).reset_index()

output_file_path = "../../data/generator/data/mimic/mimic-iii.csv"
grouped_data.to_csv(output_file_path, index=False)
structured_format(output_file_path, output_file_path)

df = pd.read_csv(output_file_path, nrows = 500000)
print(len(df))
filtered_df = df[df['TEXT'].str.len() > 5] 
print(len(filtered_df))
# Can further filter the dataframe to sample only one note per patient ID (for DP-training)
# filtered_df = df.groupby("SUBJECT_ID").first().reset_index()
filtered_df.to_csv(output_file_path) """
# STEP: Creating datasets for the n-most frequent ICD9-CODEs to be used for training the generator.
output_file_path = "../../data/generator/data/mimic/mimic-iii.csv"
args.data.path_to_dataset = output_file_path
dataset = data_utils.ALL_DATASETS[args.data.dataset_name](args, tokenizer = None)

data = dataset.create_dataset('../../data/generator/data/mimic/mimic-iii-3', is_top_freq = 3)
data = dataset.create_dataset('../../data/generator/data/mimic/mimic-iii-5', is_top_freq = 5)
data = dataset.create_dataset('../../data/generator/data/mimic/mimic-iii-10', is_top_freq = 10)
