""" Metrics for the evaluation steps """

import logging
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
from scipy.stats import stats
from sklearn.metrics import roc_auc_score
from transformers import EvalPrediction

from src.data.utils import DatasetInfo
from src.visualization.tensorboard_utils import MySummaryWriter

logger = logging.getLogger(__name__)


def _generic_compute_metrics_binary_clf(preds_clf_logits: torch.tensor,
                                        labels_clf: torch.tensor,
                                        label_col: str,
                                        metrics_dict: Dict,
                                        using_sigmoid: bool):
    """
        Internal function to compute all metrics for binary classification
    :param preds_clf_logits:
    :param labels_clf:
    :param label_col:
    :param metrics_dict:
    :return:
    """
    assert len(preds_clf_logits.shape) == 1, preds_clf_logits.shape

    if using_sigmoid:
        assert len(preds_clf_logits.shape) == 1, preds_clf_logits.shape  # Batch size * num_class
        preds_clf_probs = torch.sigmoid(preds_clf_logits)
        preds_clf = preds_clf_probs >= 0.5
    else:
        assert len(preds_clf_logits.shape) == 2, preds_clf_logits.shape  # Batch size * num_class
        preds_clf_probs = torch.softmax(preds_clf_logits, dim=-1)[:, 1]
        preds_clf = preds_clf_probs >= 0.5

    tp = ((preds_clf == 1) & (labels_clf == 1)).float().sum(0)
    fp = ((preds_clf == 1) & (labels_clf == 0)).float().sum(0)
    tn = ((preds_clf == 0) & (labels_clf == 0)).float().sum(0)
    fn = ((preds_clf == 0) & (labels_clf == 1)).float().sum(0)

    accuracy = ((tp + tn) / (tp + fp + tn + fn))
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    f1 = 2 * (precision * recall) / (precision + recall)
    # MCC formula https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()

    metrics_dict[label_col + "_ACCURACY"] = np.nan_to_num(accuracy.numpy())
    metrics_dict[label_col + "_PRECISION"] = np.nan_to_num(precision.numpy())
    metrics_dict[label_col + "_RECALL"] = np.nan_to_num(recall.numpy())
    metrics_dict[label_col + "_F1"] = np.nan_to_num(f1.numpy())
    metrics_dict[label_col + "_MCC"] = np.nan_to_num(mcc.numpy())
    metrics_dict[label_col + "_AUC"] = roc_auc_score(y_true=labels_clf.numpy(),
                                                     y_score=preds_clf_probs.numpy())
    return metrics_dict


def _generic_compute_metrics_multiclass_clf(preds_clf_logits: torch.Tensor,
                                            labels_clf: torch.Tensor,
                                            metrics_dict: Dict,
                                            label_col: str):
    """
        Internal function to compute all metrics for multiclass classification
    :param preds_clf_logits:
    :param labels_clf:
    :param metrics_dict:
    :return:
    """

    assert (labels_clf.shape[-1] == 1) and (preds_clf_logits.shape[-1] > 1)
    preds_clf_probs = torch.softmax(preds_clf_logits, dim=-1)

    accuracy = (preds_clf_probs.argmax(-1).eq(labels_clf.reshape(-1))).float().mean()

    metrics_dict[label_col + "_ACCURACY"] = np.nan_to_num(accuracy.numpy())

    return metrics_dict


def _generic_compute_metrics_regression(preds: torch.tensor,
                                        labels: torch.tensor,
                                        metrics_dict: Dict,
                                        label_col: str):
    """
        Internal function to compute all metrics for regression
    :param preds:
    :param labels:
    :param metrics_dict:
    :return:
    """

    assert len(preds.shape) == 2, preds.shape  # Batch size * num_class
    assert (labels.shape[-1] == 1) and (preds.shape[-1] == 1)

    spearman_corr = stats.spearmanr(preds.reshape(-1), labels.reshape(-1))[0]
    person_corr = stats.pearsonr(preds.reshape(-1), labels.reshape(-1))[0]

    metrics_dict[label_col + "_SPEARMAN_CORR"] = np.nan_to_num(spearman_corr)
    metrics_dict[label_col + "_PERSON_CORR"] = np.nan_to_num(person_corr)

    return metrics_dict


@dataclass
class ComputeMetricsDocumentElectraForPretraining:
    """
        Class to compute metrics for pretraining tasks.
        This class doesn't fit the HuggingFace's Trainer because it accepts list of Tensor for predictions and labels.
        The reason is to allow variable lengths predictions (ex: number of tokens is not equal to number of sentences).
    """

    def __init__(self,
                 tb_writer: MySummaryWriter,
                 hparams: Dict,
                 sentence_predictions: bool):
        self.tb_writer = tb_writer
        self.hparams = hparams
        if sentence_predictions:
            self.label_binary_col = ["is_fake_token", "is_most_corrupted_sentences"]
        else:
            self.label_binary_col = ["is_fake_token"]
        self.label_multi_col = ["mlm", "sampled_mlm"]

    def __call__(self, predictions: List[torch.Tensor], labels: List[torch.Tensor]):

        metrics_dict = {}

        # Compute metrics for binary classification
        for i, label_col in enumerate(self.label_binary_col):
            if predictions[i].shape[0] != 0:
                metrics_dict = _generic_compute_metrics_binary_clf(preds_clf_logits=predictions[i],
                                                                   labels_clf=labels[i],
                                                                   label_col=label_col,
                                                                   metrics_dict=metrics_dict,
                                                                   using_sigmoid=True
                                                                   )

        # Compute metrics for multiclass classification
        for i, label_col in enumerate(self.label_multi_col):
            i_label = i + len(self.label_binary_col)
            accuracy = (predictions[i_label].eq(labels[i_label])).sum().float() / labels[i_label].shape[0]
            metrics_dict[label_col + "_ACCURACY"] = np.nan_to_num(accuracy.numpy())

        # Workaround to fix the issue with the hparams feature in TensorBoard
        self.tb_writer.add_hparams(hparam_dict=self.hparams,
                                   metric_dict=metrics_dict,
                                   launched_by_huggingface=False)

        return {k: v.item() for k, v in metrics_dict.items()}


@dataclass
class ComputeMetricsDocumentElectraForDownstream:
    """
        Class to compute metrics for downstream tasks.
        Unlike the pretraining task, this class fits with the signature for the HuggingFace's Trainer class,
        meaning it receives Tensor for predictions and labels.
    """

    tb_writer: MySummaryWriter
    hparams: Dict
    dataset_info: DatasetInfo

    def __call__(self, eval_predictions: EvalPrediction):
        preds = torch.tensor(eval_predictions.predictions)
        labels = torch.tensor(eval_predictions.label_ids)

        # Shape correction if only one label
        if len(preds.shape) == 1:
            preds = preds[:, None]
        if len(labels.shape) == 1:
            labels = labels[:, None]

        assert len(preds.shape) == 2, preds.shape  # Docs * labels
        assert len(labels.shape) == 2, labels.shape  # Docs * labels

        metrics_dict = {}

        if self.dataset_info.is_regression:
            # Regression
            metrics_dict = _generic_compute_metrics_regression(preds=preds,
                                                               labels=labels,
                                                               label_col="label",
                                                               metrics_dict=metrics_dict
                                                               )
        else:
            # Compute metrics for classification
            output_shape = self.dataset_info.num_clf_classes
            if output_shape == 2:
                # Binary classification
                metrics_dict = _generic_compute_metrics_binary_clf(preds_clf_logits=preds[:, 0],
                                                                   labels_clf=labels[:, 0],
                                                                   label_col="label",
                                                                   metrics_dict=metrics_dict,
                                                                   using_sigmoid=True
                                                                   )
            else:
                # Multiclass classification
                metrics_dict = _generic_compute_metrics_multiclass_clf(preds_clf_logits=preds,
                                                                       labels_clf=labels,
                                                                       label_col="label",
                                                                       metrics_dict=metrics_dict
                                                                       )

        # Workaround to fix the issue with the hparams feature in TensorBoard
        self.tb_writer.add_hparams(hparam_dict=self.hparams,
                                   metric_dict=metrics_dict,
                                   launched_by_huggingface=False)

        return {k: v.item() for k, v in metrics_dict.items()}
