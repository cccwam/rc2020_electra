""" Command line entry point to train a tokenizer model"""

import logging

import fire

from src.models.train_model_tokenizer import train_tokenizer

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    fire.Fire(train_tokenizer)
