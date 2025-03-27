from dataclasses import dataclass, field
from typing import Optional
    
@dataclass
class BaseArgs:
    source_text_column: str = field(
        default='source', metadata={"help": "Name of the column containing source texts."}
    )
    ref_text_column: str = field(
        default='reference', metadata={"help": "Name of the column containing reference texts."}
    )
    device_id: int = field(
        default=0, metadata={"help": "ID of the device to use for computation."}
    )
    output_pkl_file_path: str = field(
        default='', metadata={"help": "The pickle file where the results from the evaluation are saved."}
    )
    def __post_init__(self):
        if self.output_pkl_file_path is None:
            self.output_pkl_file_path = self.default_output_pkl_file_path()

    def default_output_pkl_file_path(self) -> str:
        return 'results/result.pkl'

@dataclass
class MauveArgs(BaseArgs):
    """
    Arguments for calculating MAUVE scores.
    """
    model_name_featurizer: str = field(
        default='gpt2', metadata={"help": "Name of the model to use for feature extraction."}
    )
    max_text_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum length of the text."}
    )
    verbose: bool = field(
        default=False, metadata={"help": "Whether to print detailed logs during computation."}
    )
    def default_putput_pkl_file_path(self) -> str:
        return 'results/mauve_source.pkl'

@dataclass
class FrechetArgs(BaseArgs):
    """
    Arguments for calculating FID scores.
    """
    sent_transformer_model_name: str = field(
        default='all-MiniLM-L6-v2', metadata={"help": "Name of the model to use for feature extraction."}
    )
    def default_putput_pkl_file_path(self) -> str:
        return 'results/frechet-results.pkl'

@dataclass
class LMArgs(BaseArgs):
    """
    Arguments for calculating LM-based metric scores.
    """
    model_name: str = field(
        default='gpt2', metadata={"help": "Name of the model to use for feature extraction."}
    )
    def default_putput_pkl_file_path(self) -> str:
        return 'results/lm-metrics-results.pkl'
    
@dataclass
class Arguments:
    frechet:FrechetArgs
    mauve:MauveArgs
    perplexity:LMArgs