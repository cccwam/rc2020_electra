""" Utility for Tensorboard."""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from transformers import is_torch_available


def to_sanitized_dict(self) -> Dict[str, Any]:
    """
    Sanitized serialization to use with TensorBoard’s hparams
    This method is strongly inspired by the HuggingFace's method
    See transformers\training_args.py
    """
    d = self.__dict__.copy()
    valid_types = [bool, int, float, str]
    if is_torch_available():
        valid_types.append(torch.Tensor)
    return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


def get_tensorboard_experiment_id(experiment_name: str, tensorboard_tracking_folder: str):
    """
    Create a unique id for TensorBoard for the experiment
    :param experiment_name: name of experiment
    :param tensorboard_tracking_folder: Path where to store TensorBoard data and save trained model
    """
    model_sub_folder = experiment_name + "-" + datetime.utcnow().isoformat().replace(":", "-")
    return str(Path(tensorboard_tracking_folder) / model_sub_folder)


class MySummaryWriter(SummaryWriter):
    """
        Custom Summary writer
    """
    file_writer_hparams = None

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None,
                    launched_by_huggingface=True):
        """

        Same as default method except it includes a workaround to prevent huggingface to write hparams (unable to
        disable it without touching their training loop) and also to have a workaround to fix the issue with hparams
        in pytorch / tensorboard interface

        """
        if launched_by_huggingface:  # Hack to prevent HuggingFace to write hparams
            return

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')

        # AUC metrics to be displayed by default
        sorted_metrics = list([k for k in metric_dict.keys() if "AUC" in k])  #
        sorted_metrics += sorted(list([k for k in metric_dict.keys() if "AUC" not in k]))
        sorted_metrics = {"eval_" + k: 1 for k in sorted_metrics}

        # Display most important first by default
        sorted_hparams = ["learning_rate",
                          "pretrain_path",
                          "num_hidden_layers"]  # Most important metrics
        sorted_hparams += sorted([k for k in hparam_dict.keys() if k not in sorted_hparams])
        sorted_hparams = {k: hparam_dict[k] for k in sorted_hparams}

        exp, ssi, sei = hparams(sorted_hparams, sorted_metrics)

        self._get_file_writer().add_summary(exp)
        self._get_file_writer().add_summary(ssi)
        self._get_file_writer().add_summary(sei)
