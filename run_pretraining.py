""" Command line entry point to run the pretraining task"""
import logging

import fire

from src.models.train_model_pretraining import train_model_pretraining

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    fire.Fire(train_model_pretraining)
