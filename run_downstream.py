""" Command line entry point to run the classification task."""
import logging

import fire

from src.models.train_model_downstream import train_model_downstream

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    fire.Fire(train_model_downstream)
