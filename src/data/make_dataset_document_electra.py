"""Dataset creation logic using Huggingface NLP library. """

import logging
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import torch
import torch.utils.data
from datasets import load_dataset, concatenate_datasets, DatasetDict
from spacy.lang.en import English
from tokenizers import Tokenizer

from src.data.utils import DatasetInfo

logger = logging.getLogger(__name__)


def _validate_dataset_infos(dataset_infos: List[DatasetInfo]):
    """
        Validate if the list is correct
    :param dataset_infos:
    :return:
    """

    assert isinstance(dataset_infos, List)
    assert len(dataset_infos) >= 1
    assert any([isinstance(dataset_info, DatasetInfo) for dataset_info in dataset_infos])

    assert all([dataset_info.is_pretraining for dataset_info in dataset_infos]) or \
           (len(dataset_infos) == 1 and dataset_infos[0].is_downstream)


def _get_cache_path(dataset_info: DatasetInfo,
                    cache_dir: str) -> Path:
    """
    :param dataset_info:
    :param cache_dir:
    :return: the path for the cache file for the processed dataset
    """
    cache_path: Path = Path(cache_dir if cache_dir else "")
    cache_path /= "data"
    cache_path /= f"{dataset_info.name}-{dataset_info.subset}.cache"

    return cache_path


def make_datasets_document_electra(tokenizer: Tokenizer,
                                   dataset_info: Union[DatasetInfo, List[DatasetInfo]],
                                   dataset_sampling: Union[float, List[float]],
                                   cache_dir: str,
                                   dataset_dir: str,
                                   validation_set_size: Optional[int] = None,
                                   test_set_size: Optional[int] = None,
                                   training_set_size: int = -1,
                                   training_set_random_seed: int = 42,
                                   valid_test_split_random_seed: int = 42,
                                   num_proc: int = None
                                   ) -> Tuple[torch.utils.data.dataset.Dataset,
                                              torch.utils.data.dataset.Dataset,
                                              torch.utils.data.dataset.Dataset]:
    """
    Create the Pytorch datasets from HuggingFace.

    :param tokenizer: Already trained tokenizer
    :param dataset_info: DatasetInfo or list of DatasetInfo to be used to create the train, val and test sets
    :param dataset_sampling:
    :param dataset_dir: Directory path for the cache for the HuggingFace datasets library
    :param cache_dir: Directory to store the cache for the processed datasets.
    :param validation_set_size: Validation set size.
    :param test_set_size: Test set size. If None, then use the dataset["test"]. Default None.
    :param training_set_size: Default -1 to use all possible samples
    :param training_set_random_seed: Seed used only for shuffle the training set
    :param valid_test_split_random_seed: Seed used only for the split between test and validation sets.
        Required to ensure the validation set remains the same if seed is used.
    :param num_proc: Number of processor for the creation of the dataset
    :return: train dataset, validation dataset, test dataset
    """

    if isinstance(dataset_info, DatasetInfo):
        dataset_info = [dataset_info]
    _validate_dataset_infos(dataset_infos=dataset_info)

    if isinstance(dataset_sampling, float):
        dataset_sampling = [dataset_sampling]

    assert len(dataset_info) == len(dataset_sampling)

    logger.info(f"Dataset preprocessing started")
    combined_datasets = []
    for d_info, d_sampling in zip(dataset_info, dataset_sampling):
        cache_path: Path = _get_cache_path(dataset_info=d_info, cache_dir=cache_dir)

        if cache_path.exists():
            logger.info(f"Load a cached dataset from {cache_path}")
            processed_datasets = DatasetDict.load_from_disk(dataset_dict_path=str(cache_path))
            logger.info(f"Load completed")
        else:
            if dataset_dir:
                logger.info(f"Load unprocessed dataset from {d_info.name} using cache {dataset_dir}")
                datasets: DatasetDict = load_dataset(d_info.name, d_info.subset,
                                                     cache_dir=f"{dataset_dir}\\.cache\\huggingface\\datasets")
            else:
                logger.info(f"Load unprocessed dataset from {d_info.name} "
                            f"{d_info.subset if d_info.subset else ''}")
                datasets: DatasetDict = load_dataset(d_info.name, d_info.subset)
            #               dataset: Dataset = load_dataset(d_info.name, d_info.subset, split="train[:1000]")  # For debug only
            #                datasets = DatasetDict()
            #                datasets["train"] = dataset
            assert isinstance(datasets, DatasetDict)
            logger.info(f"Load unprocessed dataset completed")

            # Encode the dataset
            # WARNING this step takes 8.30 for sentence segmentation + tokenization of Wikipedia dataset
            logger.info(f"Dataset preprocessing")

            datasets: DatasetDict = datasets.map(function=partial(_encode_by_batch,
                                                                  tokenizer=tokenizer,
                                                                  nlp=_get_sentence_segmentation_model(),
                                                                  dataset_info=d_info
                                                                  ),
                                                 batched=True,
                                                 num_proc=num_proc,
                                                 remove_columns=d_info.text_columns)

            processed_datasets = DatasetDict()
            processed_datasets["train"] = datasets["train"]

            if d_info.validation_set_names:
                processed_datasets["validation"] = concatenate_datasets([datasets[name]
                                                                         for name in d_info.validation_set_names])

            if d_info.test_set_names:
                processed_datasets["test"] = concatenate_datasets([datasets[name]
                                                                   for name in d_info.test_set_names])

            # Remove all features which are not required by the models
            # to allow the concatenation across different datasets
            nested_features = [v for _, v in processed_datasets.column_names.items()]
            flatten_features = [item for items in nested_features for item in items]
            extra_cols = set(flatten_features) - {"input_ids", "label"}
            processed_datasets.remove_columns_(list(extra_cols))

            logger.info(f"Cache this processed dataset into {cache_path}")
            processed_datasets.save_to_disk(dataset_dict_path=str(cache_path))
            # Workaround to force the processed_dataset to remove the extra columns
            processed_datasets = DatasetDict.load_from_disk(dataset_dict_path=str(cache_path))
            logger.info(f"Cache completed")

        if d_sampling < 1.0:
            logger.info(f"Dataset downsampling started")
            for d_name in processed_datasets:
                processed_datasets[d_name] = processed_datasets[d_name].select(
                    np.arange(int(len(processed_datasets[d_name]) * d_sampling)))
            logger.info(f"Dataset downsampling completed")

        combined_datasets += [processed_datasets]

    logger.info(f"Dataset preprocessing completed")

    train_set = [combined_dataset["train"] for combined_dataset in combined_datasets]
    if len(train_set) > 1:
        train_set = concatenate_datasets(train_set)
    else:
        train_set = train_set[0]

    val_combined_dataset = [combined_dataset["validation"] for combined_dataset in combined_datasets
                            if "validation" in combined_dataset]
    if len(val_combined_dataset) > 1:
        val_set = concatenate_datasets(val_combined_dataset)
    elif len(val_combined_dataset) == 1:
        val_set = val_combined_dataset[0]
    else:
        val_set = None

    test_combined_dataset = [combined_dataset["test"] for combined_dataset in combined_datasets
                             if "test" in combined_dataset]
    if len(test_combined_dataset) > 1:
        test_set = concatenate_datasets(test_combined_dataset)
    elif len(test_combined_dataset) == 1:
        test_set = test_combined_dataset[0]
    else:
        test_set = None

    assert len(train_set) > 0, "Your train set is empty"

    if test_set_size is None and validation_set_size is not None:
        # Case you extract a validation set from train set and no test set
        subset_indices = [validation_set_size, len(train_set) - validation_set_size]

        generator = torch.Generator().manual_seed(valid_test_split_random_seed)
        val_set, train_set = torch.utils.data.random_split(dataset=train_set,
                                                           lengths=subset_indices,
                                                           generator=generator)
    elif test_set_size is not None and validation_set_size is not None:
        # Case you extract a test and validation set from train set
        subset_indices = [test_set_size, test_set_size + validation_set_size]  # 100 for Electra in their code source
        subset_indices += [len(train_set) - sum(subset_indices)]

        generator = torch.Generator().manual_seed(valid_test_split_random_seed)
        test_set, val_set, train_set = torch.utils.data.random_split(dataset=train_set,
                                                                     lengths=subset_indices,
                                                                     generator=generator)
    else:
        # Case you don't need to use data from train set for validation and/or test sets
        train_set = train_set.shuffle(seed=training_set_random_seed)

    # Ability to use smaller dataset
    if training_set_size != -1:
        generator = torch.Generator().manual_seed(training_set_random_seed)
        train_set, _ = torch.utils.data.random_split(dataset=train_set,
                                                     lengths=[training_set_size, len(train_set) - training_set_size],
                                                     generator=generator)

    assert val_set is not None and len(val_set) > 0, "Your validation set is empty"

    logger.info(f"Training size: {len(train_set)}")
    logger.info(f"Valid size: {len(val_set)}")

    if test_set:
        logger.info(f"Test size: {len(test_set)}")

    return train_set, val_set, test_set


def _get_sentence_segmentation_model():
    """
    Util function to create the sentence segmentation model
    :return:
    """
    # Load Spacy model
    nlp = English()
    # nlp.max_length = 1000000 * 8 # Equivalent of 8 GB of memory / Required to have all text in memory
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    return nlp


def _encode_by_batch(documents: Dict[str, List],
                     tokenizer: Tokenizer,
                     nlp: English,
                     dataset_info: DatasetInfo,
                     bos_token="<BOS>",
                     sep_token="<SEP>",
                     ) -> Dict[str, List]:
    """
    Perform sentence segmentaton and tokenization.

    :param documents: List of all documents (string)
    :param tokenizer: Tokenizer
    :param bos_token: BOS token to be added
    :param sep_token: SEP token to be added
    :return:
    """
    encoded_docs = []

    assert len(dataset_info.text_columns) >= 1
    for i_d in range(len(documents[dataset_info.text_columns[0]])):
        if dataset_info.sentence_segmentation:
            assert len(dataset_info.text_columns) == 1, dataset_info
            d = [s.text for s in nlp(documents[dataset_info.text_columns[0]][i_d]).sents]
        else:
            d = [documents[text_column][i_d] for text_column in dataset_info.text_columns]

        texts = [s + sep_token for s in d]
        texts[0] = bos_token + texts[0]
        encoded_doc = [encoding.ids for encoding in tokenizer.encode_batch(texts)]
        encoded_docs += [encoded_doc]

    if "label" in documents:
        return {"input_ids": encoded_docs,
                "label": documents["label"]}
    else:
        return {"input_ids": encoded_docs}
