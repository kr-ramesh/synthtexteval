from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DescriptorArgs:
    """
    Arguments for the descriptive metrics
    """
    pkl_file_path : str = field(
        default='entity-output.pkl', metadata={"help": "Path to the file where the entity analysis results will be saved."}
    )
    plt_file_path : str = field(
        default='entity-analysis.png', metadata={"help": "Path to the file containing the plot of the entity analysis."}
    )
    min_threshold: int = field(
        default=10, metadata={"help": "Returns the n-least frequent entities."}
    )
    max_threshold: int = field(
        default=10, metadata={"help": "Returns the n-most frequent entities."}
    )
    produce_plot: bool = field(
        default=False, metadata={"help": "If set to true, a plot of the most frequent entities is generated and saved."}
    )
    device_id: int = field(
        default=0, metadata={"help": "ID of the device to use for computation."}
    )
    