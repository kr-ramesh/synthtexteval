from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TextDescriptorArgs:
    """
    Arguments for the text descriptor module.
    """
    pkl_file_path : str = field(
        default='entity-output.pkl', metadata={"help": "Indicates the path to the pickle file where the analysis of the entity results will be saved."}
    )
    plt_file_path : str = field(
        default='entity-analysis.png', metadata={"help": "Indicates the path to the png file containing the plot of the entities sorted by frequency."}
    )
    min_threshold: int = field(
        default=10, metadata={"help": "Indicates the lower bound of the threshold with which we return the n-least frequent entities."}
    )
    max_threshold: int = field(
        default=10, metadata={"help": "Indicates the upper bound of the threshold with which we return the n-most frequent entities."}
    )
    produce_plot: bool = field(
        default=False, metadata={"help": "If set to true, a plot of the most frequent entities is generated and saved. If set to false, no plot is generated."}
    )
    