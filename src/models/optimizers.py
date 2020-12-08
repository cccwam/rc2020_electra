"""Custom logics related to optimizer and learning rate scheduling."""
from typing import Tuple

import torch
from torch.optim import Optimizer
from torch.optim.adamw import AdamW
from transformers import TrainingArguments
from transformers.optimization import get_linear_schedule_with_warmup

from src.models.modeling_document_electra import DocumentElectraModelForDownstream


# noinspection PyProtectedMember,PyUnresolvedReferences,DuplicatedCode
def create_optimizers(lr,
                      layerwise_lr_decay_power,
                      model: DocumentElectraModelForDownstream,
                      training_args: TrainingArguments,
                      max_steps: int,
                      debug=False) -> Tuple[Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
        Return a tuple containing an AdamW optimizer including layerwise lr decay and a linear scheduler

    :param lr: Default learning rate
    :param layerwise_lr_decay_power: layerwise lr decay as per Electra implementation
    :param model: Model to be trained
    :param training_args: Training arguments
    :param max_steps: Max steps used for the linear scheduler
    :param debug: Default: False
    :return: Tuple containing the optimizer and the learning rate scheduler
    """
    if hasattr(model.document_electra.word_encoder, "transformer"):
        base_model_word = model.document_electra.word_encoder.transformer
    else:
        raise NotImplementedError()

    layers = base_model_word.encoder.layers if hasattr(base_model_word.encoder, "layers") \
        else base_model_word.encoder.layer
    depth = len(layers)

    # noinspection DuplicatedCode
    if hasattr(model.document_electra, "sentence_encoder"):
        if hasattr(model.document_electra.sentence_encoder, "transformer"):
            base_model_sentence = model.document_electra.sentence_encoder.transformer
        else:
            raise NotImplementedError()
        sent_layers = base_model_sentence.encoder.layers if hasattr(base_model_sentence.encoder, "layers") \
            else base_model_sentence.encoder.layer
        depth += len(sent_layers)
    else:
        base_model_sentence = None

    # Custom optimizer to have layer wise learning rate
    layerwise_lrs = [lr * layerwise_lr_decay_power ** layer for layer in range(depth)][::-1]
    #    return

    # Embedding layers have got the same learning rate as per first layer
    params_groups = [{"params": base_model_word.embeddings,
                      "lr": layerwise_lrs[0]},
                     {"params": model.document_electra.word_encoder.token_embeddings,
                      "lr": layerwise_lrs[0]},
                     {"params": model.document_electra.word_encoder.sequence_embeddings,
                      "lr": layerwise_lrs[0]},
                     {"params": model.document_electra.word_encoder.relative_position_embeddings,
                      "lr": layerwise_lrs[0]},
                     {"params": model.document_electra.word_encoder.embeddings_project,
                      "lr": layerwise_lrs[0]},
                     ]
    # Layerwise learning rate
    params_groups += [{"params": layer, "lr": layerwise_lrs[i]}
                      for i, layer in enumerate(layers)]
    # Final linear layers have got the same learning rate as per last layer
    offset = len(layers)
    params_groups += [{"params": model.document_electra.word_encoder_final_linear,
                       "lr": layerwise_lrs[offset - 1]}]

    if base_model_sentence:
        # Embedding layers have got the same learning rate as per first layer
        params_groups += [{"params": base_model_sentence.embeddings,
                           "lr": layerwise_lrs[offset]},
                          {"params": model.document_electra.sentence_encoder.token_embeddings,
                           "lr": layerwise_lrs[offset]},
                          {"params": model.document_electra.sentence_encoder.sequence_embeddings,
                           "lr": layerwise_lrs[offset]},
                          {"params": model.document_electra.sentence_encoder.relative_position_embeddings,
                           "lr": layerwise_lrs[offset]},
                          {"params": model.document_electra.sentence_encoder.embeddings_project,
                           "lr": layerwise_lrs[offset]}]
        # Layerwise learning rate
        params_groups += [{"params": layer, "lr": layerwise_lrs[offset + i]}
                          for i, layer in enumerate(base_model_sentence.encoder.layers)]
        # Final linear layers have got the same learning rate as per last layer
        params_groups += [{"params": model.document_electra.sentence_encoder_final_linear,
                           "lr": layerwise_lrs[-1]}]

    if debug:
        print([{"params": params_group["params"], "lr": params_group["lr"]} for params_group in params_groups])

    # We add the head on the params_groups to have the largest LR reported in TensorBoard and WanDB
    # Note: Default HuggingFace's Trainer loop take the first params_group
    params_groups = [{"params": model.head,
                      "lr": layerwise_lrs[-1]}] + params_groups

    optimizer = AdamW(
        params=[{"params": params_group["params"].parameters(), "lr": params_group["lr"]}
                for params_group in params_groups],
        lr=lr,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon
    )

    assert max_steps > 0
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps,
        num_training_steps=max_steps
    )

    return optimizer, lr_scheduler
