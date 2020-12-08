""" Utility functions."""

from dataclasses import dataclass
from typing import Dict, Any

import torch
from tokenizers import Tokenizer
from transformers import TrainingArguments, set_seed

from src.visualization.tensorboard_utils import to_sanitized_dict


def get_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    Helper function to load Huggingface tokenizers
    :param tokenizer_path:
    :return: tokenizer instance
    """

    tokenizer = Tokenizer.from_file(path=tokenizer_path)

    tokenizer.add_special_tokens([
        "<PAD>",  # Padding values must be 0
        "<MASK>",  # Masked tokens must be 1
        "<BOS>",  # BOS must be 2
        "<EOS>",  # EOS must be 3
        "<SEP>",  # SEP must be 4
        "<UNK>",  # UNK must be 5 (not relevant for BBPE but still present in vocab)
    ])

    tokenizer.no_truncation()

    return tokenizer


@dataclass
class MyTrainingArguments(TrainingArguments):
    """
        Class to overload the default HF TrainingArguments in order to keep custom parameters in Tensorboard
    """
    pretrain_path: str = None
    freeze_weight: bool = None
    optimizer: str = None
    training_set_size: int = None
    layerwise_lr_decay_power: float = None

    training_set_random_seed: int = None
    valid_test_split_random_seed: int = None

    threshold_to_train_generator: float = None
    metric_to_train_generator: str = None

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        This method is strongly inspired by the HuggingFace's method
        See transformers\training_args.py
        """
        return to_sanitized_dict(self)


# noinspection PyUnresolvedReferences
def my_set_seed(seed: int):
    """

    :param seed:
    """
    set_seed(seed=seed)
    torch.backends.cudnn.deterministic = True  # needed for reproducible experiments
    torch.backends.cudnn.benchmark = False
