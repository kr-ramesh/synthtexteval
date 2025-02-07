from dataclasses import dataclass, field
from typing import Optional

#TODO: Create common argument base class for the mutual arguments

@dataclass
class MauveArgs:
    """
    Arguments for calculating MAUVE scores.
    
    Attributes:
        source_text_column (str): Name of the column containing source texts (default: 'source').
        ref_text_column (str): Name of the column containing reference texts (default: 'reference').
        device_id (int): ID of the device to use for computation (default: 0).
        model_name_featurizer (str): Name of the model to use for feature extraction (default: 'gpt2').
        max_test_length (Optional[int]): Maximum length of the test texts (default: None).
        verbose (bool): Whether to print detailed logs during computation (default: False).
    """
    source_text_column: str = field(
        default='source', metadata={"help": "Name of the column containing source texts."}
    )
    ref_text_column: str = field(
        default='reference', metadata={"help": "Name of the column containing reference texts."}
    )
    device_id: int = field(
        default=0, metadata={"help": "ID of the device to use for computation."}
    )
    model_name_featurizer: str = field(
        default='gpt2', metadata={"help": "Name of the model to use for feature extraction."}
    )
    max_text_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum length of the text."}
    )
    verbose: bool = field(
        default=False, metadata={"help": "Whether to print detailed logs during computation."}
    )
    output_pkl_file_path: str = field(
        default='results/mauve-results.pkl', metadata={"help": "The pickle file where the results from the evaluation are saved."}
    )

@dataclass
class FrechetArgs:
    """
    Arguments for calculating FID scores.
    
    Attributes:
        source_text_column (str): Name of the column containing source texts (default: 'source').
        ref_text_column (str): Name of the column containing reference texts (default: 'reference').
        device_id (int): ID of the device to use for computation (default: 0).
        model_name_featurizer (str): Name of the model to use for feature extraction (default: 'gpt2').
        max_test_length (Optional[int]): Maximum length of the test texts (default: None).
        verbose (bool): Whether to print detailed logs during computation (default: False).
    """
    source_text_column: str = field(
        default='source', metadata={"help": "Name of the column containing source texts."}
    )
    ref_text_column: str = field(
        default='reference', metadata={"help": "Name of the column containing reference texts."}
    )
    device_id: int = field(
        default=0, metadata={"help": "ID of the device to use for computation."}
    )
    sent_transformer_model_name: str = field(
        default='all-MiniLM-L6-v2', metadata={"help": "Name of the model to use for feature extraction."}
    )
    output_pkl_file_path: str = field(
        default='results/frechet-results.pkl', metadata={"help": "The pickle file where the results from the evaluation are saved."}
    )

@dataclass
class LMArgs:
    """
    Arguments for calculating LM-based metric scores.
    
    Attributes:
        source_text_column (str): Name of the column containing source texts (default: 'source').
        ref_text_column (str): Name of the column containing reference texts (default: 'reference').
        model_name (str): Name of the model to use for perplexity-based evals.
    """
    source_text_column: str = field(
        default='source', metadata={"help": "Name of the column containing source texts."}
    )
    model_name: str = field(
        default='gpt2', metadata={"help": "Name of the model to use for feature extraction."}
    )
    output_pkl_file_path: str = field(
        default='results/lm-metrics-results.pkl', metadata={"help": "The pickle file where the results from the evaluation are saved."}
    )
