from dataclasses import dataclass, field
from transformers import HfArgumentParser
from utils import jaccard_similarity, lda_similarity
import pandas as pd

@dataclass
class Arguments:
    ref_data_path: str = field(metadata={
        "help": "Path to the reference data to be compared. (CSV format)"
    })
    base_data_path: str = field(metadata={
        "help": "Path to the base data to be compared. (CSV format)"
    })
    column_name: str = field(default="TEXT", metadata={
        "help": "Column name in the CSV files containing the text data to read."
    })

def main(args):
    #Comment this out if you have multiple argument objects
    args = args[0]

    print(f'Ref data directory: {args.ref_data_path}')
    print(f'Base data directory: {args.base_data_path}')
    print(f'Columns to compare: {args.column_name}')

    df1, df2 = pd.read_csv(args.ref_data_path), pd.read_csv(args.base_data_path)
    corpus1 = df1[args.column_name].tolist()
    corpus2 = df2[args.column_name].tolist()
    #corpus1, corpus2 = corpus1[:10], corpus2[:10]

    print("LDA KL Divergence:", lda_similarity(corpus1, corpus2))
    print("Jaccard Similarity:", jaccard_similarity(corpus1, corpus2))

if __name__ == '__main__':
    parser = HfArgumentParser((Arguments))
    # Parses arguments from command line
    parsed_args = parser.parse_args_into_dataclasses()
    main(args = parsed_args)
