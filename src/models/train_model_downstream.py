"""Entry point for the classification task."""
import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict, Optional

import torch
from transformers import is_wandb_available

if is_wandb_available():
    from transformers.integrations import WandbCallback
    from src.visualization.wandb_callbacks import MyWandbCallback

from transformers.trainer import Trainer
from transformers.trainer_utils import EvaluationStrategy

from src.data.make_dataset_document_electra import make_datasets_document_electra
from src.data.utils import get_dataset_info
from src.features.features_document_electra import DataCollatorForDocumentElectra
from src.models.metrics import ComputeMetricsDocumentElectraForDownstream
from src.models.modeling_document_electra import DocumentElectraConfig, DocumentElectraModelForDownstream
from src.models.optimizers import create_optimizers
from src.models.utils import get_tokenizer, MyTrainingArguments, my_set_seed
from src.visualization.tensorboard_utils import get_tensorboard_experiment_id, MySummaryWriter, to_sanitized_dict

logger = logging.getLogger(__name__)

_default_settings = {}
if torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory < 2 ** 30 * 6):
    # Less than 6 GB GPU memory
    _default_settings["max_length"] = 128 * 16  # 2048
    _default_settings["max_position_embeddings"] = 4096
    _default_settings["per_device_train_batch_size"] = 32
    _default_settings["num_hidden_layers"] = 12
    _default_settings["effective_batch_size"] = 32
    _default_settings["lr"] = 3e-4
else:
    _default_settings["max_length"] = 128 * 16  # 2048
    _default_settings["max_position_embeddings"] = 4096
    _default_settings["per_device_train_batch_size"] = 32
    _default_settings["num_hidden_layers"] = 12
    _default_settings["effective_batch_size"] = 32
    _default_settings["lr"] = 3e-4

import os



def train_model_downstream(
        dataset_name: str = "imdb",
        dataset_subset: str = None,
        dataset_dir: str = None,
        cache_dir: str = None,
        output_dir: str = "checkpoints",
        tokenizer_path: str = r"models/ByteLevelBPETokenizer-vocab_size=30522-min_frequency=2",
        gradient_accumulation_steps: int = (_default_settings["effective_batch_size"] /
                                            _default_settings["per_device_train_batch_size"]),
        # 1 for ElectraSmall but aggregation across docs decrease variance
        per_device_train_batch_size: int = _default_settings["per_device_train_batch_size"],
        per_device_eval_batch_size: int = _default_settings["per_device_train_batch_size"],

        lr: int = _default_settings["lr"],  # 5e-4 for ElectraSmall * effective batch size

        epochs: int = 3,

        # 10 epochs
        max_grad_norm: float = 1,  # 1 for Electra
        # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py
        embedding_size: int = 128,  # Output and input embedding size, 128 for ElectraSmall
        hidden_size: int = 256,  # 256 for ElectraSmall
        chunk_length: int = 128,
        num_hidden_layers: int = _default_settings["num_hidden_layers"],  # 12 for ElectraSmall
        num_attention_heads: int = 4,
        layer_depth_offset: int = -1,

        max_sentence_length: int = 128,
        max_sentences: int = 128,
        max_length: int = _default_settings["max_length"],
        max_position_embeddings: int = _default_settings["max_position_embeddings"],

        intermediate_size: int = 1024,  # 1024 for ElectraSmall
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,  # 0.1 for ElectraSmall
        attention_probs_dropout_prob: float = 0.1,  # 0.1 for ElectraSmall
        logging_steps: int = _default_settings["effective_batch_size"] * 10,  # Default logging every 5 grad updates
        eval_steps: int = 0.5,

        seed: int = 42,
        training_set_random_seed: int = 42,
        valid_test_split_random_seed: int = 42,
        training_set_size: int = -1,
        num_proc: int = 0,
        gradient_checkpointing: bool = False,  # Decrease mem by 5/10% but increase compute cost
        pretrain_path: str = None,
        warmup_steps: Union[int, float] = 0.1,  # 10% for ElectraSmall
        weight_decay: float = 0.00,  # 0.00 for ElectraSmall
        tensorboard_tracking_folder: str = "tensorboard",
        freeze_weight: bool = False,
        layerwise_lr_decay_power: float = 0.8,  # Same as Electra for finetuning
        fcn_dropout: float = 0.1,  # 0.1 for Electra
        relative_position_embeddings: bool = True,
        sequence_embeddings: bool = True,
        experiment_name: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Main function to launch downstream tasks
    :param experiment_name: Name of experiment
    :param dataset_name: HuggingFace NLP dataset. Default Wikipedia
    :param dataset_subset: Default '20200501.en'
    :param dataset_dir: Directory path for the cache for the HuggingFace datasets library
    :param cache_dir: Directory to store the cache for the processed datasets.
    :param output_dir: Checkpoints folder
    :param tokenizer_path: Path to tokenizer
    :param gradient_accumulation_steps: Default 1.
    :param per_device_train_batch_size: Batch size per GPU. Default 32.
    :param per_device_eval_batch_size: Batch size per GPU. Default 32.
    :param lr: Learning rate. Default 5e-4 like in Electra paper * 32
    :param max_grad_norm: Gradient norm clipping. Default 1 like Electra
    :param embedding_size: Output embedding size for the Electra encoder
    :param hidden_size: Hidden embedding size within encoders. Default 256 like ElectraSmall
    :param num_hidden_layers: Number of layer. Default 12 to be equivalent to ElectraSmall (12)
    :param num_attention_heads: Number of attention heads. Default 4 like ElectraSmall
    :param chunk_length: Default 128.
    :param layer_depth_offset: Define which layer to use as sentence embedding or document embedding. Default -1.

    :param max_sentence_length: Longer sentences will be truncated.
    :param max_sentences: Longer document will be truncated
    :param max_length: maximum size for a document. It could be less than max_sequence_length * max_sentences.
    If None, then the max_length will be max_sequence_length * max_sentences.
    Default None
    :param max_position_embeddings: Maximum allowed sequence length for the model. Default 4096.

    :param intermediate_size: Default 1024 like ElectraSmall
    :param hidden_act: Activation for encoder. Default "gelu" like Electra
    :param hidden_dropout_prob: Dropout probability for FCN layers for encoder.

    :param epochs: Number of epochs for training.
    :param logging_steps: Logging steps (show train loss) in number of samples (and not in gradient updates)
    :param eval_steps: Evaluate steps (evaluate and show val loss) in number of samples (and not in gradient updates).
    If a float is provided, then the evaluation step is every eval_steps * steps per epoch.

    :param weight_decay: Default 0.00 like ElectraSmall
    :param warmup_steps: Default 10% like ElectraSmall
    :param num_proc: Number of processor for data preprocessing steps
    :param training_set_size: Default -1 to use all possible training set.
    :param seed: Seed used in everything except for dataset split and shuffling
    :param training_set_random_seed: Seed used only for shuffle the training set
    :param valid_test_split_random_seed: Seed used only for the split between test and validation sets.
        Required to ensure the validation set remains the same if seed is used.
    :param attention_probs_dropout_prob:
    :param tensorboard_tracking_folder:
    :param pretrain_path: Path to the pretrained checkpoints. Model weights and optimizer states will be loaded.
            This will allow to continue training.
    :param gradient_checkpointing: Default False

    :param freeze_weight: Freeze weight for the encoder. Default False
    :param layerwise_lr_decay_power: Layerwise LR decay. Default 0.8 as per Electra paper
    :param fcn_dropout: Dropout probability for before the final linear layer. Default 0.2 unlike 0.1 for ElectraSmall
    compared to the discriminant at token level (only for num_layers). Default 25%

    :param relative_position_embeddings: Use relative position embeddings. Default True
    :param sequence_embeddings: Use sequence embeddings (number of sentences). Default True
    :return:
    """
    my_set_seed(seed=seed)

    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)

    dataset_info = get_dataset_info(dataset_name=dataset_name, dataset_subset=dataset_subset)
    assert dataset_info.is_downstream

    if experiment_name is None:
        experiment_name = f"{dataset_name}{'-' + dataset_subset if dataset_subset else ''}"

    config = DocumentElectraConfig(vocab_size=tokenizer.get_vocab_size(),
                                   hidden_act=hidden_act,
                                   hidden_dropout_prob=hidden_dropout_prob,
                                   attention_probs_dropout_prob=attention_probs_dropout_prob,
                                   embedding_size=embedding_size,
                                   hidden_size=hidden_size,
                                   chunk_length=chunk_length,
                                   intermediate_size=intermediate_size,
                                   num_attention_heads=num_attention_heads,
                                   num_hidden_layers=num_hidden_layers,
                                   layer_depth_offset=layer_depth_offset,
                                   max_sentence_length=max_sentence_length,
                                   max_sentences=max_sentences,
                                   max_position_embeddings=max_position_embeddings,
                                   max_length=max_length,
                                   gradient_checkpointing=gradient_checkpointing,
                                   class_output_shape=(dataset_info.num_clf_classes if dataset_info.is_classification
                                                       else dataset_info.num_regr),
                                   regr_output_shape=dataset_info.num_regr,
                                   relative_position_embeddings=relative_position_embeddings,
                                   sequence_embeddings=sequence_embeddings,
                                   fcn_dropout=fcn_dropout
                                   )
    # model = DocumentElectraModelForDownstream(config=config, tokenizer=tokenizer)  # Only for debug
    model = DocumentElectraModelForDownstream(config=config)

    if pretrain_path is not None:
        pretrain_path = Path(pretrain_path)
        assert pretrain_path.exists() and pretrain_path.is_dir()
        pretrain_path /= "pytorch_model.bin"
        logger.info(f"Load pretrained weights from {pretrain_path}")
        pretrain_dict = torch.load(str(pretrain_path))
        pretrain_dict = OrderedDict([(
            k.replace("discriminant.document_electra", "document_electra"), v)
            for k, v in pretrain_dict.items() if k.startswith(
                "discriminant.document_electra")])
        model.load_state_dict(state_dict=pretrain_dict, strict=False)
        logger.info(f"Load pretrained weights from {pretrain_path} Completed")

    if freeze_weight:
        # Freeze the base model
        logger.info(f"Freeze weights for pretrained Document Electra layers")
        for params in model.document_electra.parameters():
            params.requires_grad = False

    train_set, val_set, test_set = make_datasets_document_electra(
        tokenizer=tokenizer,
        dataset_info=dataset_info,
        dataset_sampling=[1.0],  # No downsampling
        cache_dir=cache_dir,
        dataset_dir=dataset_dir,
        training_set_size=training_set_size,
        training_set_random_seed=training_set_random_seed,
        valid_test_split_random_seed=valid_test_split_random_seed,
        num_proc=2)  # Multiple processors will need a dataset copy (so it increase the disk space requirement)

    train_data_collator = DataCollatorForDocumentElectra(config=config)

    effective_train_batch_size = per_device_train_batch_size * gradient_accumulation_steps

    # Workaround: Bug in Huggingface if the training size is not a multiple of gradient accumulation steps
    # noinspection PyTypeChecker
    num_update_steps_per_epoch = int(len(train_set) // effective_train_batch_size)
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    max_steps = math.ceil(epochs * num_update_steps_per_epoch)

    # noinspection PyTypeChecker
    training_args = MyTrainingArguments(
        output_dir=get_tensorboard_experiment_id(experiment_name=experiment_name,
                                                 tensorboard_tracking_folder=output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,

        save_steps=max(int(len(train_set) / effective_train_batch_size * 1.0), 1),  # Save every 100%
        save_total_limit=20,

        learning_rate=lr,
        fp16=True,
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir=get_tensorboard_experiment_id(experiment_name=experiment_name,
                                                  tensorboard_tracking_folder=tensorboard_tracking_folder),
        logging_steps=int(logging_steps / effective_train_batch_size),
        eval_steps=(int(eval_steps / effective_train_batch_size) if type(eval_steps) == int
                    else int(eval_steps * (len(train_set) / effective_train_batch_size))),
        do_eval=True,
        do_train=True,
        evaluation_strategy=EvaluationStrategy.STEPS,
        seed=seed,
        # Custom attributes to keep in TB
        pretrain_path=str(pretrain_path),
        freeze_weight=freeze_weight,
        layerwise_lr_decay_power=layerwise_lr_decay_power,
        optimizer="AdamW",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,  # If lower, this may create an issue for amp.
        warmup_steps=(int(max_steps * warmup_steps) if isinstance(warmup_steps, float) else warmup_steps),
        weight_decay=weight_decay,
        dataloader_num_workers=num_proc,
        training_set_size=training_set_size,
        training_set_random_seed=training_set_random_seed,
        valid_test_split_random_seed=valid_test_split_random_seed,
        label_names=["labels"]
    )

    # Instantiate TensorBoard
    # Required to be done outside Trainer in order to log in TensorBoard via custom metrics (for non scalar logging)
    # And also to keep more hparams in TensorBoard
    tb_writer = MySummaryWriter(log_dir=training_args.logging_dir)
    tb_writer.add_text("config", config.to_json_string())
    tb_writer.add_text("args", training_args.to_json_string())
    tb_writer.add_hparams({**to_sanitized_dict(config),
                           **to_sanitized_dict(training_args),
                           }, metric_dict={})

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=train_data_collator,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=ComputeMetricsDocumentElectraForDownstream(tb_writer=tb_writer,
                                                                   dataset_info=dataset_info,
                                                                   hparams={**to_sanitized_dict(config),
                                                                            **to_sanitized_dict(training_args),
                                                                            }),
        optimizers=create_optimizers(lr=lr,
                                     layerwise_lr_decay_power=layerwise_lr_decay_power,
                                     model=model,
                                     training_args=training_args,
                                     max_steps=max_steps),
        tb_writer=tb_writer
    )

    if is_wandb_available():
        # Workaround to force the creation of a new Wandb callback for each run if we launches several run
        trainer.pop_callback(WandbCallback)
        trainer.add_callback(MyWandbCallback)

    trainer.train()

    eval_output: Dict[str, float] = trainer.evaluate()
    logger.info(eval_output)

    return eval_output
