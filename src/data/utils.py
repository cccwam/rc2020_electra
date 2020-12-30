"""
    Class to retrieve some meta information about dataset based on name and subset.
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DatasetInfo:
    """
        Class containing meta information about dataset
    """
    name: str
    subset: Optional[str]
    text_columns: List[str]
    validation_set_names: List[str]
    test_set_names: List[str]
    sentence_segmentation: bool
    num_clf_classes: int  # Number of classification labels
    num_regr: int  # Number of regression labels

    @property
    def is_classification(self):
        """
        :return: True if classification task else False
        """
        return self.num_clf_classes > 0

    @property
    def is_regression(self):
        """
        :return: True if regression task else False
        """
        return not self.is_classification

    @property
    def is_downstream(self):
        """
        :return: True if downstream task else False
        """
        return (self.num_regr > 0) or (self.num_clf_classes > 0)

    @property
    def is_pretraining(self):
        """
        :return: True if pretraining task else False
        """
        return not self.is_downstream

    def __post_init(self):
        if self.is_downstream:
            assert (self.num_clf_classes == 0) or (self.num_regr == 0), "Only single task are allowed"


# noinspection SpellCheckingInspection
_dataset_infos: List[DatasetInfo] = [
    # glue
    DatasetInfo(name="glue", subset="cola", num_clf_classes=2, num_regr=0,
                text_columns=["sentence"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),
    DatasetInfo(name="glue", subset="sst2", num_clf_classes=2, num_regr=0,
                text_columns=["sentence"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),
    DatasetInfo(name="glue", subset="mrpc", num_clf_classes=2, num_regr=0,
                text_columns=["sentence1", "sentence2"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),
    DatasetInfo(name="glue", subset="qqp", num_clf_classes=2, num_regr=0,
                text_columns=["question1", "question2"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),
    DatasetInfo(name="glue", subset="stsb", num_clf_classes=0, num_regr=1,
                text_columns=["sentence1", "sentence2"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),
    DatasetInfo(name="glue", subset="mnli", num_clf_classes=3, num_regr=0,
                text_columns=["premise", "hypothesis"], sentence_segmentation=False,
                validation_set_names=["validation_matched", "validation_mismatched"],
                test_set_names=["test_matched", "test_mismatched"]),
    DatasetInfo(name="glue", subset="qnli", num_clf_classes=2, num_regr=0,
                text_columns=["question", "sentence"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),
    DatasetInfo(name="glue", subset="rte", num_clf_classes=2, num_regr=0,
                text_columns=["sentence1", "sentence2"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),
    DatasetInfo(name="glue", subset="wnli", num_clf_classes=2, num_regr=0,
                text_columns=["sentence1", "sentence2"], sentence_segmentation=False,
                validation_set_names=["validation"], test_set_names=["test"]),

    # IMDB
    DatasetInfo(name="imdb", subset=None, num_clf_classes=2, num_regr=0,
                text_columns=["text"], sentence_segmentation=True,
                validation_set_names=["test"], test_set_names=[]),

    # Wikipedia
    DatasetInfo(name="wikipedia", subset="20200501.en",
                num_clf_classes=0, num_regr=0,
                text_columns=["text"], sentence_segmentation=False,
                validation_set_names=[], test_set_names=[]),

    # BookCorpus
    DatasetInfo(name="bookcorpus", subset=None,
                num_clf_classes=0, num_regr=0,
                text_columns=["text"], sentence_segmentation=False,
                validation_set_names=[], test_set_names=[]),

    # OWT
    DatasetInfo(name="openwebtext", subset=None,
                num_clf_classes=0, num_regr=0,
                text_columns=["text"], sentence_segmentation=False,
                validation_set_names=[], test_set_names=[]),
]


def get_dataset_info(dataset_name: str, dataset_subset: str) -> DatasetInfo:
    """
        Return the DatasetInfo based on name and subset
    :param dataset_name:
    :param dataset_subset:
    :return:
    """
    dataset_infos = []
    for _dataset_info in _dataset_infos:
        if (_dataset_info.name == dataset_name) and ((_dataset_info.subset is None and dataset_subset is None)
                                                     or (_dataset_info.subset == dataset_subset)):
            dataset_infos += [_dataset_info]

    assert len(dataset_infos) == 1, dataset_infos
    return dataset_infos[0]
