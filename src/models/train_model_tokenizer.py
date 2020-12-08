"""Entry point to train the tokenizer."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from datasets import load_from_disk, load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import BaseTokenizer

logger = logging.getLogger(__name__)

# Hard coded to keep typing checks as the each class has got different signatures
ALGORITHM_MAPPINGS = {"ByteLevelBPETokenizer": ByteLevelBPETokenizer}


def get_unique_id(algorithm: str,
                  vocab_size: int,
                  min_frequency: int):
    """

    :param algorithm:
    :param vocab_size:
    :param min_frequency:
    :return:
    """
    return f"{algorithm}-vocab_size={vocab_size}-min_frequency={min_frequency}"


def train_tokenizer(
        output_dir: str,
        dataset: str = "imdb",
        algorithm: str = "ByteLevelBPETokenizer",
        vocab_size: int = 30522,
        min_frequency: int = 2,
        seed: int = 42,
        max_documents: int = 100000
):
    """
    Main function to train tokenizers.
    Special tokens are hard-coded as following:
        "<PAD>",  # Padding values must be 0
        "<MASK>",  # Masked tokens must be 1
        "<BOS>",  # BOS must be 2
        "<EOS>",  # EOS must be 3
        "<SEP>",  # SEP must be 4
        "<UNK>",  # UNK must be 5 (not relevant for BBPE but still present in vocab)


    :param output_dir: Path where the tokenizer will be saved
    :param dataset: Name or path for the HuggingFace nlp library
    :param algorithm: ByteLevelBPETokenizer (Electra use WordPiece as BERT)
    :param vocab_size: Default 30522 like Electra
    https://github.com/google-research/electra/blob/master/configure_pretraining.py
    :param min_frequency: Default 2
    :param seed: Default 42
    :param max_documents: If number of documents is higher, then subsampling to prevent OOM
    :return:
    """
    if algorithm not in ALGORITHM_MAPPINGS:
        raise NotImplementedError(f"Algorithm {algorithm} not yet covered")

    np.random.seed(seed)
    logger.info(f"Using seed {seed}")

    tokenizer: BaseTokenizer = ALGORITHM_MAPPINGS[algorithm]()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if Path(dataset).exists():
        logger.info(f"Dataset {dataset} will be loaded from disk")
        dataset = load_from_disk(dataset_path=dataset)
    else:
        logger.info(f"Dataset {dataset} is a standard HuggingFace dataset")
        dataset = load_dataset(dataset, split="train")
    logger.info(f"Dataset {dataset} loaded")

    max_documents = min(max_documents, len(dataset))
    logger.info(f"max_documents {max_documents}")

    # Using all files will create OOM errors
    # Better to subsample to have roughly 3GB of data (recommendation from HF in their github is 1GB)
    f = NamedTemporaryFile(mode='w+', delete=False)
    logger.info(f"Write all documents in tmp file {f.name}")
    for i in np.random.randint(low=0,
                               high=len(dataset),
                               size=max_documents):
        txt: str = dataset[int(i)]["text"]
        f.writelines([txt + "\n"])

    logger.info(f"Train tokenizer from tmp file")
    # noinspection PyUnresolvedReferences
    tokenizer.train(files=[f.name], vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
        "<PAD>",  # Padding values must be 0
        "<MASK>",  # Masked tokens must be 1
        "<BOS>",  # BOS must be 2
        "<EOS>",  # EOS must be 3
        "<SEP>",  # SEP must be 4
        "<UNK>",  # UNK must be 5 (not relevant for BBPE but still present in vocab)
    ])

    tokenizer_unique_id: str = get_unique_id(algorithm=algorithm,
                                             vocab_size=vocab_size,
                                             min_frequency=min_frequency)
    logger.info(tokenizer_unique_id)
    tokenizer.save(path=str(output_dir / tokenizer_unique_id), pretty=True)
    logger.info(f"Model saved in {output_dir}/{tokenizer_unique_id}")
