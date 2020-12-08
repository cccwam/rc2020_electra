"""Electra model replication"""

import logging
import pdb
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from tokenizers import Tokenizer
from torch import nn
from transformers.modeling_bert import BertPreTrainedModel
from transformers.activations import get_activation
from transformers.file_utils import ModelOutput
from transformers.modeling_bert import BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


def _count_parameters(model,
                      max_level: int = 7,
                      debug=False):
    """
    Utility function to display number of parameters
    Inspiration from https://stackoverflow.com/questions/48393608/pytorch-network-parameter-calculation

    :param model: PyTorch model
    :param max_level: Maximum depth level to keep during the aggregation.
    :param debug: Default False. If true, number of parameters for each layers will be displayed
    """
    df = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name = name.replace(".weight", "").replace(".bias", "").split(".")
            layer_name = {i: layer_name[i] for i in range(len(layer_name))}
            num_param = np.prod(param.size())
            layer_name["params"] = num_param
            df += [layer_name]
            if debug:
                # Display the dimension
                if param.dim() > 1:
                    print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                else:
                    print(name, ':', num_param)

    df = pd.DataFrame(df)
    # Aggregate layers depending their depth to be easier to read
    df = df.fillna("").groupby([i for i in range(max_level)]).sum()
    pd.set_option('display.max_rows', None)
    print(df)
    print(f"total : {df.params.sum()}")


class DocumentElectraConfig(PretrainedConfig):
    """
        Document Electra config class. Strongly inspired by HuggingFace BertConfig.

    """
    model_type = "document_electra"

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_hidden_layers: int,
                 num_attention_heads: int, intermediate_size: int,
                 max_sentence_length: int, max_sentences: int, max_position_embeddings: int,
                 max_length: int,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.05, attention_probs_dropout_prob: float = 0.05,
                 pad_token_id: int = 0, mask_token_id: int = 1,
                 bos_token_id: int = 2, eos_token_id: int = 3, sep_token_id: int = 4,
                 gradient_checkpointing: bool = False, generator_size: float = 0.25,
                 generator_layer_size: float = 1.0,
                 discriminant_loss_factor: float = 50, mlm_probability: float = 0.15,
                 mlm_replacement_probability: float = 0.85, temperature: float = 1.0,
                 class_output_shape: int = None, regr_output_shape: int = None,
                 fcn_dropout: float = 0.1, chunk_length: int = 128, layer_depth_offset: int = -1,
                 initializer_range: float = 0.02,
                 sequence_embeddings: bool = False, relative_position_embeddings: bool = True):
        super().__init__()
        self.sequence_embeddings = sequence_embeddings
        self.relative_position_embeddings = relative_position_embeddings
        self.layer_depth_offset = layer_depth_offset
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.return_dict = True
        self.torchscript = False

        self.chunk_length = chunk_length
        self.mask_token_id = mask_token_id
        self.sep_token_id = sep_token_id

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.max_position_embeddings = max_position_embeddings if max_position_embeddings \
            else max_sentence_length * max_sentences
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.max_length = max_length
        assert self.max_position_embeddings >= self.max_length

        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.gradient_checkpointing = gradient_checkpointing

        self.initializer_range = initializer_range

        # Downstream task specifics
        self.is_downstream_task = not (class_output_shape is None and regr_output_shape is None)
        if self.is_downstream_task:
            self.is_regression = regr_output_shape is not None and regr_output_shape > 0

            self.num_classes = class_output_shape
            assert self.num_classes is not None and self.num_classes > 0

            self.fcn_dropout = fcn_dropout
            assert self.fcn_dropout is not None

        # Pretraining task specifics
        if not self.is_downstream_task:
            self.mlm_probability = mlm_probability
            self.mlm_replacement_probability = mlm_replacement_probability
            self.temperature = temperature
            self.generator_size = generator_size
            self.generator_layer_size = generator_layer_size if generator_layer_size else generator_size
            self.discriminant_loss_factor = discriminant_loss_factor


class MyTransformerModel(BertPreTrainedModel):
    """
        Decorator on top of Transformer model to integrate own embedding logic.
    """

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: BertConfig,
                 embedding_size: int,
                 max_relative_position_ids: int,
                 max_sentence_ids: int,
                 relative_position_embeddings: bool = None,
                 sequence_embeddings: bool = None):
        super().__init__(config)
        self.config = config

        self.transformer = BertModel(config, add_pooling_layer=False)
        # noinspection PyTypeChecker
        self.transformer.set_input_embeddings(None)  # We use our own token embeddings

        self.sequence_embeddings = sequence_embeddings
        self.relative_position_embeddings = relative_position_embeddings

        assert self.relative_position_embeddings is not None
        assert self.sequence_embeddings is not None
        if self.relative_position_embeddings:
            self.relative_position_embeddings = nn.Embedding(max_relative_position_ids, embedding_size)
        if self.sequence_embeddings:
            self.sequence_embeddings = nn.Embedding(max_sentence_ids, embedding_size)
        self.token_embeddings = nn.Embedding(config.vocab_size, embedding_size)

        if embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(embedding_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            sequence_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None) -> BaseModelOutputWithPooling:
        """

        :param input_ids:
        :param attention_mask:
        :param position_ids:
        :param sequence_ids:
        :param head_mask:
        :param inputs_embeds:
        :param output_hidden_states:
        :param output_attentions:
        :param return_dict:
        :return:
        """
        assert (input_ids is None and inputs_embeds is not None) | (input_ids is not None and inputs_embeds is None)

        assert position_ids is not None
        assert sequence_ids is not None

        # No attention on padding
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            else:
                attention_mask = inputs_embeds.byte().any(-1).ne(self.config.pad_token_id)
            assert len(attention_mask.shape) == 2

        assert input_ids is not None
        inputs_embeds = self.token_embeddings.forward(input=input_ids)
        if self.relative_position_embeddings:
            inputs_embeds += self.relative_position_embeddings.forward(input=position_ids)
        if self.sequence_embeddings:
            inputs_embeds += self.sequence_embeddings.forward(input=sequence_ids)

        if hasattr(self, "embeddings_project") and (inputs_embeds is not None):
            inputs_embeds = self.embeddings_project.forward(inputs_embeds)

        # noinspection PyArgumentEqualDefault
        return self.transformer.forward(
            input_ids=None,  # We use our own embedding
            position_ids=None,  # The model will also add absolute position embeddings,
            token_type_ids=None,  # We use our own
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, return_dict=return_dict)


class DocumentElectraPreTrainedModel(PreTrainedModel, torch.nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    config_class = DocumentElectraConfig
    config: DocumentElectraConfig  # Force the right type
    base_model_prefix = "document_electra"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """

        :param pretrained_model_name_or_path:
        :param model_args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _pad_inputs(self, input_ids, position_ids, sequence_ids, sequence_length):
        if sequence_length % self.config.chunk_length != 0:
            padding = (sequence_length // self.config.chunk_length + 1) * self.config.chunk_length
            padding -= sequence_length
            input_ids = torch.nn.functional.pad(input=input_ids,
                                                pad=[0, padding],
                                                mode='constant',
                                                value=self.config.pad_token_id)
            position_ids = torch.nn.functional.pad(input=position_ids,
                                                   pad=[0, padding],
                                                   mode='constant',
                                                   value=self.config.pad_token_id)
            sequence_ids = torch.nn.functional.pad(input=sequence_ids,
                                                   pad=[0, padding],
                                                   mode='constant',
                                                   value=self.config.pad_token_id)
        return input_ids, position_ids, sequence_ids

    def _pad_embeddings(self, sentence_embeddings):
        if sentence_embeddings.shape[1] % self.config.chunk_length != 0:
            padding = (sentence_embeddings.shape[1] // self.config.chunk_length + 1) * self.config.chunk_length
            padding -= sentence_embeddings.shape[1]
            sentence_embeddings = torch.nn.functional.pad(input=sentence_embeddings,
                                                          pad=[0, 0,
                                                               0, padding],
                                                          mode='constant',
                                                          value=self.config.pad_token_id)
        return sentence_embeddings


@dataclass
class DocumentElectraModelModelOutput:
    """
        Output for the DocumentElectraModel
    """
    pretraining_word_embeddings: torch.Tensor
    downstream_word_embeddings: torch.Tensor
    pretraining_sentence_embeddings: Optional[torch.Tensor] = None
    downstream_sentence_embeddings: Optional[torch.Tensor] = None


# noinspection PyAbstractClass
class DocumentElectraModel(DocumentElectraPreTrainedModel):
    """
        Document Electra Base model.

        2 hierarchical encoders: one to get word level embeddings, and a second one to use the BOS embeddings as input_ids
        and to get a new sentence level embeddings.

        Strongly inspired by HuggingFace library.

    """

    # noinspection PyPep8
    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        # Hierarchical encoder
        config_word_encoder = BertConfig(hidden_act=config.hidden_act,
                                         hidden_dropout_prob=config.hidden_dropout_prob,
                                         attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                         vocab_size=config.vocab_size, hidden_size=config.hidden_size,
                                         feed_forward_size=config.intermediate_size,
                                         intermediate_size=config.intermediate_size,
                                         num_hidden_layers=config.num_hidden_layers,
                                         num_attention_heads=config.num_attention_heads,
                                         max_position_embeddings=config.max_position_embeddings + 1,
                                         chunk_size_feed_forward=0, pad_token_id=config.pad_token_id,
                                         bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                         sep_token_id=config.sep_token_id,
                                         initializer_range=config.initializer_range,
                                         output_hidden_states=True,
                                         return_dict=True,
                                         gradient_checkpointing=config.gradient_checkpointing,
                                         torchscript=False)

        self.word_encoder = MyTransformerModel(config=config_word_encoder,
                                               embedding_size=config.embedding_size,
                                               relative_position_embeddings=config.relative_position_embeddings,
                                               sequence_embeddings=config.sequence_embeddings,
                                               max_relative_position_ids=config.max_sentence_length + 1,
                                               max_sentence_ids=config.max_sentences + 1)
        hidden_size = config.hidden_size
        self.word_encoder_final_linear = nn.Linear(hidden_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
    ):
        """

        :param input_ids:
        :param position_ids:
        :param sequence_ids:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs, sequence_length = input_ids.shape[0], input_ids.shape[1]

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        input_ids, position_ids, sequence_ids = self._pad_inputs(input_ids, position_ids, sequence_ids, sequence_length)

        # prediction_scores at word level
        outputs = self.word_encoder.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_ids=sequence_ids)

        ###
        #   Embedding at token levels
        ###

        # Compute embedding for the pretraining task at token level
        pretraining_word_embeddings = self.word_encoder_final_linear.forward(
            input=outputs.last_hidden_state[:, :sequence_length, :])
        assert pretraining_word_embeddings.shape == (num_docs, sequence_length, self.config.hidden_size), \
            pretraining_word_embeddings.shape

        # Retrieve the embedding for the downstream task at token level
        downstream_word_embeddings = outputs.hidden_states[self.config.layer_depth_offset][:, :sequence_length, :]
        assert downstream_word_embeddings.shape == (num_docs, sequence_length, self.config.hidden_size), \
            (downstream_word_embeddings.shape, sequence_length)

        return DocumentElectraModelModelOutput(pretraining_word_embeddings=pretraining_word_embeddings,
                                               downstream_word_embeddings=downstream_word_embeddings)


    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Module`): A module mapping vocabulary to hidden states.
        """
        self.word_encoder.set_input_embeddings(value)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


@dataclass
class DocumentElectraDiscriminatorModelOutput(ModelOutput):
    """
        Output class for the DocumentElectraDiscriminatorModel
    """
    loss: torch.tensor
    token_level_loss: Optional[float] = None
    is_fake_logits: Optional[torch.Tensor] = None


class DocumentElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: DocumentElectraConfig):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # input like config.embedding_size?
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.activation = get_activation(config.hidden_act)
        self.config = config

    def forward(self, discriminator_hidden_states):
        """

        :param discriminator_hidden_states:
        :return:
        """
        assert len(discriminator_hidden_states.shape) == 3, discriminator_hidden_states.shape

        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = self.dense_prediction(hidden_states)

        return logits


# noinspection PyAbstractClass
class DocumentElectraDiscriminatorModel(DocumentElectraPreTrainedModel):
    """
        Document Electra Discriminator model.

    """

    # noinspection PyPep8
    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        # Hierarchical encoder
        self.document_electra = DocumentElectraModel(config=config)

        # https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/run_pretraining.py#L190
        self.discriminator_electra = DocumentElectraDiscriminatorPredictions(config)
        self.discriminator_document_electra = DocumentElectraDiscriminatorPredictions(config)

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            labels_at_token_level=None,
    ):
        """

        :param input_ids:
        :param position_ids:
        :param sequence_ids:
        :param labels_at_token_level:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs, sequence_lengths = input_ids.shape

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        outputs = self.document_electra.forward(input_ids=input_ids,
                                                position_ids=position_ids,
                                                sequence_ids=sequence_ids)

        # Binary classification task at token level
        is_fake_logits = self.discriminator_electra.forward(
            discriminator_hidden_states=outputs.pretraining_word_embeddings).reshape(num_docs, sequence_lengths)
        assert is_fake_logits.shape == (num_docs, sequence_lengths), is_fake_logits.shape

        unmasked_tokens = labels_at_token_level.ne(-100)

        token_level_loss = self.loss(input=is_fake_logits[unmasked_tokens],
                                     target=labels_at_token_level[unmasked_tokens].float())

        return DocumentElectraDiscriminatorModelOutput(
            loss=token_level_loss,
            token_level_loss=token_level_loss.item(),
            is_fake_logits=is_fake_logits)


    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


@dataclass
class DocumentElectraGeneratorModelOutput(ModelOutput):
    """
        Output class for the DocumentElectraGeneratorModel
    """
    loss: torch.tensor
    documents_logits: torch.Tensor = None


class DocumentElectraGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers + linear layer (unlike HuggingFace)."""

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: DocumentElectraConfig):
        super().__init__()

        hidden_size = int(config.hidden_size * config.generator_size)
        self.dense = nn.Linear(hidden_size, config.embedding_size)
        self.activation = get_activation(config.hidden_act)
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, generator_hidden_states):
        """

        :param generator_hidden_states:
        :return:
        """
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return self.generator_lm_head(hidden_states)


# noinspection PyAbstractClass
class DocumentElectraGeneratorModel(DocumentElectraPreTrainedModel):
    """
        Document Electra Generator model.

        Strongly inspired by HuggingFace library.

    """

    # noinspection PyPep8
    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        config_sentence_encoder = BertConfig(hidden_dropout_prob=config.hidden_dropout_prob,
                                             attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                             vocab_size=config.vocab_size,
                                             hidden_size=int(config.hidden_size * config.generator_size),
                                             feed_forward_size=int(config.intermediate_size *
                                                                   config.generator_size),
                                             intermediate_size=int(config.intermediate_size *
                                                                   config.generator_size),
                                             num_hidden_layers=max(1, int(config.num_hidden_layers *
                                                                          config.generator_layer_size)),
                                             num_attention_heads=max(1,
                                                                     int(config.num_attention_heads *
                                                                         config.generator_size)),
                                             max_position_embeddings=config.max_position_embeddings,
                                             chunk_size_feed_forward=0, pad_token_id=config.pad_token_id,
                                             bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                             sep_token_id=config.sep_token_id,
                                             initializer_range=config.initializer_range,
                                             gradient_checkpointing=config.gradient_checkpointing,
                                             return_dict=True,
                                             torchscript=False)
        self.generator_encoder = MyTransformerModel(config=config_sentence_encoder,
                                                    relative_position_embeddings=config.relative_position_embeddings,
                                                    sequence_embeddings=config.sequence_embeddings,
                                                    max_relative_position_ids=config.max_sentence_length + 1,
                                                    embedding_size=config.embedding_size,
                                                    max_sentence_ids=config.max_sentences + 1)

        self.generator_predictions = DocumentElectraGeneratorPredictions(config)

        self.loss = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            labels_generator=None,  # (docs, seq)
    ):
        """

        :param input_ids:
        :param position_ids:
        :param sequence_ids:
        :param labels_generator:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        (num_docs, sequence_length) = input_ids.shape

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        assert labels_generator is not None
        assert labels_generator.shape == input_ids.shape

        input_ids, position_ids, sequence_ids = self._pad_inputs(input_ids, position_ids, sequence_ids, sequence_length)

        logits = self.generator_encoder.forward(input_ids=input_ids,
                                                position_ids=position_ids,
                                                sequence_ids=sequence_ids).last_hidden_state

        assert len(logits.shape) == 3
        assert logits.shape[0] == num_docs
        assert logits.shape[2] == int(self.config.hidden_size * self.config.generator_size), logits.shape

        logits = self.generator_predictions.forward(logits)
        logits = logits[:, :sequence_length, :]  # Remove padding
        assert logits.shape == (num_docs, sequence_length, self.config.vocab_size)

        loss = self.loss(input=logits.reshape(-1, self.config.vocab_size),
                         target=labels_generator.reshape(-1))

        return DocumentElectraGeneratorModelOutput(loss=loss, documents_logits=logits)

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        return self.generator_encoder.get_input_embeddings()

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


# noinspection PyAbstractClass
class DocumentElectraPretrainingModel(DocumentElectraPreTrainedModel):
    """
        Electra model (with generator) for Electra pretraining task as per Electra paper.

        2 models:
        - A generator
        - A discriminator

        Strongly inspired by HuggingFace library.

    """

    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        self.generator = DocumentElectraGeneratorModel(config=self.config)
        self.discriminant = DocumentElectraDiscriminatorModel(config=self.config)

        # Model extension if same size Section 3.2 of Electra paper
        # if self.config.generator_size == 1:
        # Weight sharing between generator and discriminator for input embeddings
        self.generator.generator_encoder.relative_position_embeddings = self.discriminant.document_electra.word_encoder.relative_position_embeddings
        self.generator.generator_encoder.token_embeddings = self.discriminant.document_electra.word_encoder.token_embeddings
        self.generator.generator_encoder.sequence_embeddings = self.discriminant.document_electra.word_encoder.sequence_embeddings

        # Weight sharing between input and output embedding
        self.generator.generator_predictions.generator_lm_head.weight.data = self.generator.generator_encoder.token_embeddings.weight.data  # .transpose(0,1)

        self.train_generator = True

        self.init_weights()

        _count_parameters(self)

    def _mask_tokens(self, input_ids: torch.Tensor, position_ids: torch.Tensor, sequence_ids: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        """

        assert len(input_ids.shape) == 2  # Batch size (nb of docs) * nb of tokens
        num_docs = input_ids.shape[0]

        assert position_ids.shape == input_ids.shape
        assert sequence_ids.shape == input_ids.shape

        max_sequences = sequence_ids.max()

        # Determine the most corrupted sentences for later use
        probability_matrix = torch.full(size=(num_docs, max_sequences),
                                        device=input_ids.device,
                                        fill_value=0.0)  # max_sequences
        probability_matrix = torch.bernoulli(probability_matrix)
        masked_sentence_ids = probability_matrix.bool().nonzero(as_tuple=False).flatten(start_dim=1)

        labels_at_sentence_level = torch.zeros(size=(num_docs, max_sequences),
                                               device=input_ids.device)
        labels_at_sentence_level[masked_sentence_ids[:, 0], masked_sentence_ids[:, 1]] = 1.

        # Determine the masked tokens (without additional noise from corrupted sentences)
        probability_matrix = torch.full(size=input_ids.shape,
                                        device=input_ids.device,
                                        fill_value=self.config.mlm_probability)

        # Loop of docs and masked sentence ids.
        for mask_idx in masked_sentence_ids:
            probability_matrix[mask_idx[0], sequence_ids[mask_idx[0], :].eq(mask_idx[1] + 1)] = \
                0.0
        masked_tokens = torch.bernoulli(probability_matrix).bool()

        # Create default labels (1 for fake, 0 for original, -100 for padding)
        labels_at_token_level = torch.zeros_like(input_ids)

        # Ignore special tokens
        padding_mask = input_ids.eq(self.config.pad_token_id)  # Padding value
        bos_mask = input_ids.eq(self.config.sep_token_id) | input_ids.eq(self.config.bos_token_id)
        labels_at_token_level[padding_mask | bos_mask] = -100  # We don't compute loss for these
        masked_tokens &= ~(padding_mask | bos_mask)  # And we don't corrupt them

        # Set labels for generator as -100 (ignore) for not masked tokens
        labels_generator = input_ids.clone()
        labels_generator[~masked_tokens] = -100

        # Corrupt the selected inputs with prob
        probability_matrix = torch.full(size=input_ids.shape,
                                        device=input_ids.device,
                                        fill_value=self.config.mlm_replacement_probability)
        replaced_tokens = masked_tokens & torch.bernoulli(probability_matrix).bool()

        generator_input_ids = input_ids.clone()
        generator_input_ids[replaced_tokens] = self.config.mask_token_id

        # Set labels to 1 for tokens to be replaced
        labels_at_token_level[replaced_tokens] = 1

        return (generator_input_ids, labels_generator, masked_tokens, replaced_tokens,
                labels_at_token_level, labels_at_sentence_level)

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            loss_only: bool = True
    ):

        """

        :param loss_only:
        :param sequence_ids:
        :param position_ids:
        :param input_ids:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs = input_ids.shape[0]

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        (generator_input_ids, labels_generator, masked_tokens, replaced_tokens,
         labels_at_token_level, labels_at_sentence_level) = self._mask_tokens(input_ids=input_ids,
                                                                              position_ids=position_ids,
                                                                              sequence_ids=sequence_ids)

        assert labels_generator is not None
        assert len(labels_generator.shape) == 2

        assert labels_at_token_level is not None
        assert len(labels_at_token_level.shape) == 2

        assert labels_at_sentence_level is not None
        assert len(labels_at_sentence_level.shape) == 2

        # Generator step
        if self.train_generator:
            generator_outputs = self.generator.forward(input_ids=generator_input_ids,
                                                       position_ids=position_ids,
                                                       sequence_ids=sequence_ids,
                                                       labels_generator=labels_generator)
        else:
            with torch.no_grad():
                generator_outputs = self.generator.forward(input_ids=generator_input_ids,
                                                           position_ids=position_ids,
                                                           sequence_ids=sequence_ids,
                                                           labels_generator=labels_generator)
        mlm_tokens = torch.softmax(generator_outputs.documents_logits, dim=-1).argmax(-1)
        mlm_input_ids = torch.where(replaced_tokens, mlm_tokens, input_ids)
        mlm_input_ids = mlm_input_ids.detach()  # Stop gradients

        # Sampling step
        # Using Gumbel sampling https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        sampled_tokens = torch.nn.functional.gumbel_softmax(logits=generator_outputs.documents_logits,
                                                            tau=self.config.temperature,
                                                            hard=True).argmax(-1)

        assert len(sampled_tokens.shape) == 2  # docs, tokens
        assert sampled_tokens.shape[0] == num_docs
        sampled_input_ids = torch.where(replaced_tokens, sampled_tokens, input_ids)
        sampled_input_ids = sampled_input_ids.detach()  # Stop gradients
        del sampled_tokens

        # Set labels to false when generators give the true values  # the logic should here not before
        labels_at_token_level = torch.zeros_like(input_ids)
        labels_at_token_level[
            sampled_input_ids.ne(input_ids) & labels_generator.ne(-100) & replaced_tokens] = 1  # 0 if original, else 1
        padding_mask = input_ids.eq(self.config.pad_token_id)  # Padding value
        bos_mask = input_ids.eq(self.config.sep_token_id) | input_ids.eq(self.config.bos_token_id)
        labels_at_token_level[padding_mask | bos_mask] = -100  # We don't compute loss for these

        # Discriminant step
        disc_outputs = self.discriminant.forward(input_ids=sampled_input_ids,
                                                 position_ids=position_ids,
                                                 sequence_ids=sequence_ids,
                                                 labels_at_token_level=labels_at_token_level,)

        # Combine losses
        loss = generator_outputs.loss + self.config.discriminant_loss_factor * disc_outputs.loss

        # Prepare outputs. Note: adaptation for PEP8
        output = DocumentElectraPretrainingModelOutput(loss=loss,
                                                       generator_loss=generator_outputs.loss.item(),
                                                       discriminant_loss=disc_outputs.loss.item(),
                                                       discriminant_token_loss=disc_outputs.token_level_loss)
        if not loss_only:
            output.mlm_input_ids = mlm_input_ids
            output.sampled_input_ids = sampled_input_ids
            output.labels_generator = labels_generator
            output.is_fake_logits = disc_outputs.is_fake_logits
            output.labels_at_token_level = labels_at_token_level

        return output

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


@dataclass
class DocumentElectraPretrainingModelOutput(ModelOutput):
    """
        Output class for the DocumentElectraPretrainingModel
    """
    loss: torch.tensor

    generator_loss: float = None
    mlm_input_ids: Optional[torch.Tensor] = None
    sampled_input_ids: Optional[torch.Tensor] = None
    labels_generator: Optional[torch.Tensor] = None

    discriminant_loss: Optional[float] = None
    discriminant_token_loss: Optional[float] = None
    is_fake_logits: Optional[torch.Tensor] = None
    labels_at_token_level: Optional[torch.Tensor] = None


# noinspection PyAbstractClass
class DocumentElectraModelHead(DocumentElectraPreTrainedModel):
    """
        Head on top of DocumentElectra
    """

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation(config.hidden_act)
        self.linear_2 = nn.Linear(config.hidden_size, config.num_classes)

        self.dropout = nn.Dropout(p=self.config.fcn_dropout)

    def forward(
            self,
            document_embeddings
    ):
        """

        :param document_embeddings:
        :return:
        """
        document_embeddings = self.dropout(document_embeddings)
        document_embeddings = self.linear_1.forward(document_embeddings)
        document_embeddings = self.activation(document_embeddings)
        return self.linear_2.forward(document_embeddings)


# noinspection PyAbstractClass
class DocumentElectraModelForDownstream(DocumentElectraPreTrainedModel):
    """
    Hierarchical Electra model and linear head for downstream task.

    """

    def __init__(self, config: DocumentElectraConfig, tokenizer: Tokenizer = None):
        super().__init__(config)

        self.document_electra = DocumentElectraModel(config=config)
        self.head = DocumentElectraModelHead(config=config)

        self.loss = nn.MSELoss() if self.config.is_regression else nn.CrossEntropyLoss()

        self.tokenizer = tokenizer

        self.init_weights()

        _count_parameters(self)

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            labels=None,  # (docs)
    ):
        """

        :param sequence_ids:
        :param position_ids:
        :param input_ids:
        :param labels:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs, sequence_lengths = input_ids.shape

        assert labels is not None
        assert labels.shape == torch.Size([num_docs]), (labels.shape, num_docs)

        # Document Electra encoder
        outputs = self.document_electra.forward(input_ids=input_ids,
                                                position_ids=position_ids,
                                                sequence_ids=sequence_ids)

        if outputs.downstream_sentence_embeddings is None:
            # Merge all sentences embeddings for a document embedding
            # Use the embedding for position ids == 1
            # document_embeddings = []
            # for d in range(num_docs):
            #    document_embeddings += [outputs.downstream_word_embeddings[d, position_ids[d].eq(1), :].mean(0)]
            # document_embeddings = torch.stack(document_embeddings)
            # document_embeddings = outputs.downstream_word_embeddings[:, 0, :]
            non_padding_mask = input_ids.ne(self.config.pad_token_id)
            document_embeddings = (non_padding_mask.unsqueeze(-1) * outputs.pretraining_word_embeddings).sum(
                1)  # Average pooling on all words
            document_embeddings /= non_padding_mask.sum(-1).unsqueeze(-1)
            assert document_embeddings.shape == (num_docs, self.config.hidden_size), document_embeddings.shape

        else:
            # Merge all sentences embeddings for a document embedding
            # Use the first embedding like Electra for classification
            # https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/modeling.py#L254
            document_embeddings = outputs.downstream_sentence_embeddings[:, 0, :]
            assert document_embeddings.shape == (num_docs, self.config.hidden_size)

        # Classification head
        documents_logits = self.head.forward(document_embeddings=document_embeddings)
        assert documents_logits.shape == (num_docs, self.config.num_classes), documents_logits.shape

        if self.tokenizer:
            # Debug only
            print(
                self.tokenizer.decode_batch([input_ids[0].cpu().numpy().tolist()], skip_special_tokens=False)[0].split(
                    "<PAD>")[0])
            print(labels[0])
            print(torch.softmax(documents_logits[0], dim=-1).argmax(-1),
                  torch.softmax(documents_logits[0], dim=-1),
                  documents_logits.shape)

        # Compute loss
        if self.config.is_regression and self.config.num_classes == 1:
            labels = labels.reshape(-1, 1)  # Batch size * 1
            assert documents_logits.shape == labels.shape
        loss = self.loss.forward(input=documents_logits, target=labels)

        # If binary class, keep only the logits for class 1
        if not self.config.is_regression and self.config.num_classes == 2:
            documents_logits = documents_logits[:, 1]

        return DocumentElectraModelForClassificationOutput(loss=loss,
                                                           logits=documents_logits)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


@dataclass
class DocumentElectraModelForClassificationOutput(ModelOutput):
    """
        Output class for DocumentElectraModelForDownstream
    """
    loss: torch.Tensor
    logits: Optional[torch.Tensor] = None
