from dataclasses import dataclass, field
from typing import Optional

MODEL_TYPES = ['longformer']
@dataclass
class CorefArgs:
    output_dir: str = field(metadata={"help": "The output directory where the model checkpoints and predictions will be written."})
    model_type: str = field(default="longformer", metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)})
    base_model_name_or_path: str = field(default="allenai/longformer-base-4096", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training file. If a data dir is specified, will look for the file there" + "If no data dir or train/predict files are specified, will run with tensorflow_datasets."})
    test_file: Optional[str] = field(default=None, metadata={"help": "The input testing file. If a data dir is specified, will look for the file there" + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",})
    predict_file: Optional[str] = field(default=None, metadata={"help": "The input evaluation file. If a data dir is specified, will look for the file there" + "If no data dir or train/predict files are specified, will run with tensorflow_datasets."})
    predict_file_write: Optional[str] = field(default=None, metadata={"help": "The input evaluation file. If a data dir is specified, will look for the file there" + "If no data dir or train/predict files are specified, will run with tensorflow_datasets."})
    config_name: str = field(default="allenai/longformer-base-4096", metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: str = field(default="allenai/longformer-base-4096", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store pre-trained models downloaded from s3."})
    max_seq_length: int = field(default=-1, metadata={"help": "Maximum sequence length."})
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_infer: bool = field(default=False, metadata={"help": "Whether to run inference."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run evaluation."})
    do_lower_case: bool = field(default=False, metadata={"help": "Set if using an uncased model."})
    nonfreeze_params: Optional[str] = field(default=None, metadata={"help": "Named parameters to update while training (separated by ,). The rest will kept frozen. If None or empty - train all"})
    learning_rate: float = field(default=1e-5, metadata={"help": "Initial learning rate for Adam."})
    head_learning_rate: float = field(default=3e-4, metadata={"help": "Initial learning rate for the head."})
    dropout_prob: float = field(default=0.3, metadata={"help": "Dropout probability."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Steps to accumulate before backward update."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay value."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 parameter for Adam optimizer."})
    adam_beta2: float = field(default=0.98, metadata={"help": "Beta2 parameter for Adam optimizer."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon parameter for Adam optimizer."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total training epochs."})
    warmup_steps: int = field(default=0, metadata={"help": "Number of warmup steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X steps."})
    eval_steps: int = field(default=500, metadata={"help": "Evaluate every X steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Disable CUDA if set."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite the output directory if set."})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training."})
    amp: bool = field(default=False, metadata={"help": "Use automatic mixed precision instead of 32-bit."})
    fp16_opt_level: str = field(default="O1", metadata={"help": "Apex AMP optimization level."})
    max_span_length: int = field(default=30, metadata={"help": "Maximum span length."})
    top_lambda: float = field(default=0.4, metadata={"help": "Top lambda value."})
    max_total_seq_len: int = field(default=3500, metadata={"help": "Maximum total sequence length."})
    experiment_name: str = field(default=None, metadata={"help": "Experiment name."})
    normalise_loss: bool = field(default=False, metadata={"help": "Normalize loss if set."})
    ffnn_size: int = field(default=3072, metadata={"help": "Size of FFNN."})
    save_if_best: bool = field(default=False, metadata={"help": "Save if best checkpoint found."})
    batch_size_1: bool = field(default=False, metadata={"help": "Use batch size 1 if set."})
    tensorboard_dir: str = field(default = '/tb', metadata={"help": "Directory for TensorBoard logs."})
    conll_path_for_eval: str = field(default=None, metadata={"help": "Path to CoNLL file for evaluation."})

def set_default_coref_args(base_model_dir = '/base_pretrained_model/', output_dir = './temp', test_file = "test.i2b2.jsonlines"):
    args = CorefArgs(output_dir = output_dir, predict_file = output_dir + '/silver.jsonlines', 
                    predict_file_write = output_dir + '/silver.pred.jsonlines', test_file = test_file,
                    base_model_name_or_path = base_model_dir, 
                    tokenizer_name = 'allenai/longformer-large-4096',
                    do_infer = True, num_train_epochs = 3, logging_steps = 100, 
                    save_steps=1000, eval_steps = 150, max_seq_length = 4000, 
                    normalise_loss = True, max_total_seq_len=4000, experiment_name = 'test-run', 
                    warmup_steps = 5600, adam_epsilon = 1e-6,
                    head_learning_rate = 1e-5, save_if_best = True, 
                    top_lambda = 0.4, tensorboard_dir = output_dir + '/tb')

    return args