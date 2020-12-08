"""Class to prepare inputs from dataset """
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from src.models.modeling_document_electra import DocumentElectraConfig


@dataclass
class DataCollatorForDocumentElectra:
    """
    Data collator used for document electra.

    The logic is
    - If labels, create a tensor from them
    - Pad all sentence inputs from all documents to create one tensor
    """

    config: DocumentElectraConfig

    def __call__(self, documents: List[Dict[str, List[List[int]]]]) -> Dict[str, torch.Tensor]:
        inputs = self._tensorize_batch(documents)

        for attribute in ["input_ids", "position_ids", "sequence_ids"]:
            assert attribute in inputs.keys(), f"Missing attribute {attribute} in {inputs.keys()}"
            # Batch size (nb of docs) * nb of tokens (all sentences)
            assert len(inputs[attribute].shape) == 2, inputs[attribute].shape
            assert inputs[attribute].shape[1] <= self.config.max_sentence_length * self.config.max_sentences

        if self.config.is_downstream_task:
            labels = torch.stack([torch.tensor(d["label"]) for d in documents])
            assert len(labels.shape) == 1  # Batch size (nb of docs)
            inputs["labels"] = labels

        return inputs

    def _tensorize_batch(self, documents: List[Dict[str, List[List[int]]]]) -> Dict[str, torch.Tensor]:
        inputs = {"input_ids": [],
                  "position_ids": [],
                  "sequence_ids": []}

        # Tensorize and truncation
        tmp_input_ids: List[List[torch.Tensor]] = [[torch.LongTensor(s[:self.config.max_sentence_length][:-1] +
                                                                     [self.config.sep_token_id])
                                                    for i, s in enumerate(d["input_ids"]) if
                                                    i < self.config.max_sentences]
                                                   for d in documents]

        # Loop over documents
        for input_ids in tmp_input_ids:
            sequence_ids = [[i] * len(s) for i, s in enumerate(input_ids)]
            # Flat out and keep 0 for padding
            sequence_ids = [[item + 1 for sublist in sequence_ids for item in sublist]][0]
            sequence_ids = torch.tensor(sequence_ids)

            position_ids = [list(range(len(s))) for i, s in enumerate(input_ids)]
            # Flat out and keep 0 for padding
            position_ids = [[item + 1 for sublist in position_ids for item in sublist]][0]
            position_ids = torch.tensor(position_ids)

            input_ids = [[item for sublist in input_ids for item in sublist]][0]  # Flat out
            input_ids = torch.tensor(input_ids)

            # Truncate to max length
            if self.config.max_length and input_ids.shape[0] > self.config.max_length:
                # We randomly pick a segment within the document
                offset = torch.randint(low=0, high=input_ids.shape[0] - self.config.max_length, size=[1])
                input_ids = input_ids[offset:offset + self.config.max_length]
                if offset > 0:
                    input_ids[0] = self.config.bos_token_id
                input_ids[-1] = self.config.sep_token_id
                position_ids = position_ids[:self.config.max_length]
                sequence_ids = sequence_ids[:self.config.max_length]

            inputs["input_ids"] += [input_ids]
            inputs["position_ids"] += [position_ids]
            inputs["sequence_ids"] += [sequence_ids]

        # Padding across all documents
        for attribute in ["input_ids", "position_ids", "sequence_ids"]:
            inputs[attribute] = pad_sequence(
                inputs[attribute],
                batch_first=True,
                padding_value=self.config.pad_token_id)

        return inputs
