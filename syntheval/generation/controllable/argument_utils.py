"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
Modified from : https://github.com/microsoft/dp-transformers/
"""
from typing import Optional
import numpy as np
from scipy import optimize
from transformers import TrainingArguments as HfTrainingArguments
from transformers import IntervalStrategy, logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from datasets.utils import disable_progress_bar
from prv_accountant import Accountant
from peft import LoraConfig

logger = logging.get_logger(__name__)

#TODO: Need to figure out the control code formatting confusion.

@dataclass
class PrivacyArguments:
    """
    Arguments for differentially private training.
    """
    per_sample_max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "Max per sample clip norm"})
    noise_multiplier: Optional[float] = field(default=None, metadata={"help": "Noise multiplier for DP training"})
    target_epsilon: Optional[float] = field(default=8, metadata={
        "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"
    })
    target_delta: Optional[float] = field(default=1e-5, metadata={
        "help": "Target delta, defaults to 1/N"
    })
    disable_dp: bool = field(default=False, metadata={
        "help": "Disable DP training."
    })

    def initialize(self, sampling_probability: float, num_steps: int, num_samples: int) -> None:
        if self.target_delta is None:
            self.target_delta = 1.0/num_samples
        logger.info(f"The target delta is set to be: {self.target_delta}")

        # Set up noise multiplier
        print(f"Sampling probability: {sampling_probability}")
        if self.noise_multiplier is None:
            self.noise_multiplier = find_noise_multiplier(
                sampling_probability=sampling_probability,
                num_steps=num_steps,
                target_delta=self.target_delta,
                target_epsilon=self.target_epsilon
            )
        logger.info(f"The noise multiplier is set to be: {self.noise_multiplier}")

    @property
    def is_initialized(self) -> bool:
        return (
            self.per_sample_max_grad_norm is not None and
            self.noise_multiplier is not None and
            self.target_delta is not None
        )

    def __post_init__(self):
        if self.disable_dp:
            logger.warning("Disabling differentially private training...")
            self.noise_multiplier = 0.0
            self.per_sample_max_grad_norm = float('inf')
            self.target_epsilon = None
        else:
            if bool(self.target_epsilon) == bool(self.noise_multiplier):
                raise ValueError("Exactly one of the arguments --target_epsilon and --noise_multiplier must be used.")
            if self.per_sample_max_grad_norm is None:
                raise ValueError("DP training requires --per_sample_max_grad_norm argument.")


@dataclass
class TrainingArguments(HfTrainingArguments):
    """
    Arguments for training the generator model.
    """
    dry_run: bool = field(
        default=False,
        metadata={"help": "Option for reducing training steps (2) and logging intervals (1) for quick sanity checking of arguments."})
    
    dry_test_run: bool = field(
        default=False,
        metadata={"help": "Option for reducing testing part of the pipeline for quick sanity checking of arguments."})


    def __post_init__(self):
        super().__post_init__()
        if self.dry_run:
            logger.warning("--dry_run was specified. Reducing number of training steps to 2 and logging intervals to 1...")
            self.logging_steps = 4
            self.logging_strategy = IntervalStrategy.STEPS
            self.eval_steps = None
            self.evaluation_strategy = IntervalStrategy.STEPS
            self.max_steps = 2

        if self.disable_tqdm:
            disable_progress_bar()

@dataclass
class ModelArguments:
    """
    Arguments for initializing the generator model for training and inference.
    """
    model_name: str = field(default = "gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    path_to_load_model: str = field(default = "model_directory_default", metadata={
        "help": "Path to where the model weights are saved and need to be loaded from."
    })
    path_to_save_model: str = field(default = "model_directory_default", metadata={
        "help": "Path to where the model weights are to be saved saved."
    })
    path_to_save_test_output: str = field(default = "test-outputs", metadata={
        "help": "Path to saved model output CSV file."
    })
    sequence_len: int = field(default = 1024, metadata={
        "help": "Maximum sequence length"
    })
    num_return_seq: int = field(default = 5, metadata={
        "help": "Number of generations per given input."
    })
    load_from_ckpt: bool = field(default = False, metadata={
        "help": "Load from ckpt for continual pretraining during the generator training part of the pipeline."
    })
    inference: bool = field(default=False, metadata={
        "help": "Whether or not to enable the inference part of the pipeline."
    })
    max_new_tokens: int = field(default = 1000, metadata={
        "help": "Maximum tokens to be generated"
    })
    min_new_tokens: int = field(default = 200, metadata={
        "help": "Minimum tokens to be generated"
    })
    
@dataclass
class DataArguments:
    """
    Arguments for initializing the dataset for training and inference.
    """
    dataset_name: str = field(default = "sst2", metadata={
        "help": "Dataset name in HuggingFace, e.g. 'sst2'"
    })
    path_to_dataset: str = field(default = "/data/train.csv", metadata={
        "help": "Path to the source dataset to be trained on."
    })
    path_to_test_dataset: str = field(default = "/data/test.csv", metadata={
        "help": "Path to the dataset for inference. Can be used if the dataset being tested over is outside of an existing directory."
    })
    control_field: str = field(default = "label", metadata={
        "help": "Column corresponding to the control code in the dataset."
    })
    text_field: str = field(default = "label", metadata={
        "help": "Column corresponding to the control code text field in the dataset."
    })
    label_field: str = field(default = "text", metadata={
        "help": "Column corresponding to the output text to be generated in the dataset."
    })
    prompt_begin: str = field(default = "", metadata={
        "help": "Instructions/context preceding the input in the prompt."
    })
    prompt_end: str = field(default = "", metadata={
        "help": "Instructions/context succeeding the input in the prompt."
    })
    
    
@dataclass
class LoraArguments:
    """
    Arguments for initializing the LoRA module.
    """
    enable_lora: bool = field(default=True, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=32, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    target_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        params["target_modules"] = ast.literal_eval(params["target_modules"][0])
        return LoraConfig(**params)


@dataclass
class Arguments:
    """
    Arguments for the entire pipeline.
    """
    train: TrainingArguments
    privacy: PrivacyArguments
    model: ModelArguments
    data: DataArguments
    lora: LoraArguments

def find_noise_multiplier(sampling_probability: float, num_steps: int, target_epsilon: float, target_delta: float,
                          eps_error: float=0.1) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch for Poisson sampling
    :param int num_steps: Number of optimisation steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    """
    def compute_epsilon(mu: float) -> float:
        acc = Accountant(
            noise_multiplier=mu,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=num_steps,
            eps_error=eps_error/2
        )
        return acc.compute_epsilon(num_steps)

    mu_max = 100.0

    mu_R = 1.0
    eps_R = float('inf')
    while eps_R > target_epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError("Finding a suitable noise multiplier has not converged. "
                               "Try increasing target epsilon or decreasing sampling probability.")

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)[0]

    has_converged = False
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1]-bracket[0])*0.01
        mu_guess = optimize.root_scalar(lambda mu: compute_epsilon(mu)[2]-target_epsilon, bracket=bracket, xtol=mu_err).root
        bracket = [mu_guess-mu_err, mu_guess+mu_err]
        eps_up = compute_epsilon(mu_guess-mu_err)[2]
        eps_low = compute_epsilon(mu_guess+mu_err)[0]
        has_converged = (eps_up - eps_low) < 2*eps_error
    assert compute_epsilon(bracket[1])[2] < target_epsilon + eps_error

    return bracket[1]