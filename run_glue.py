""" Command line entry point to run all GLUE tasks."""
import logging

import fire
import pandas as pd
import os

from src.models.train_model_downstream import train_model_downstream

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def run_glue(pretrain_path: str,
             experiment_name: str = None,
             seed: int = 42,
             max_length: int = 128,
             num_proc: int = 0,
             start_idx: int = 0,
             end_idx: int = None):
    """
        Orchestrator for all GLUE tasks
    :param pretrain_path:
    :param seed:
    :param experiment_name:
    :param max_length:
    :param num_proc:
    :param start_idx:
    :param end_idx:
    :return:
    """
    tasks = [
        {"subset": "cola", "metric": "eval_label_MCC", "epochs": 3},
        {"subset": "sst2", "metric": "eval_label_ACCURACY", "epochs": 3},
        {"subset": "mrpc", "metric": "eval_label_ACCURACY", "epochs": 3},
        {"subset": "stsb", "metric": "eval_label_SPEARMAN_CORR", "epochs": 10},
        {"subset": "qqp", "metric": "eval_label_ACCURACY", "epochs": 3},
        {"subset": "mnli", "metric": "eval_label_ACCURACY", "epochs": 3},
        {"subset": "qnli", "metric": "eval_label_ACCURACY", "epochs": 3},
        {"subset": "rte", "metric": "eval_label_ACCURACY", "epochs": 10},
        {"subset": "wnli", "metric": "eval_label_ACCURACY", "epochs": 3},
    ]

    if experiment_name is not None and experiment_name != "":
        experiment_name += "-"
    elif experiment_name is None:
        experiment_name = ""

    assert 0 <= start_idx < len(tasks)
    assert end_idx is None or (0 < end_idx <= len(tasks))

    all_metrics = []
    for i in range(start_idx, end_idx if end_idx else len(tasks)):
        os.environ["WANDB_PROJECT"] = f"{experiment_name}glue"

        metrics = train_model_downstream(dataset_name="glue",
                                         dataset_subset=tasks[i]["subset"],
                                         epochs=tasks[i]["epochs"],
                                         experiment_name=f"{experiment_name}glue-{tasks[i]['subset']}-seed_{seed}",
                                         pretrain_path=pretrain_path,
                                         seed=seed,
                                         max_length=max_length,
                                         num_proc=num_proc,
                                         )
        all_metrics += [metrics[tasks[i]["metric"]]]

    results = pd.DataFrame(tasks)
    results["result"] = all_metrics
    logger.info("############################ Results Per task ############################")
    logger.info(results.to_markdown())
    logger.info("##########################################################################")
    logger.info("############################ GLUE Score ############################")
    logger.info(results["result"].mean())
    logger.info("##########################################################################")


if __name__ == '__main__':
    fire.Fire(run_glue)
