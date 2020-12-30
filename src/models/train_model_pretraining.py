"""Entry point for the pretraining task."""
import collections
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Union, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from packaging import version
from torch.cuda.amp import autocast
from torch.optim import Optimizer
# noinspection PyProtectedMember
from torch.utils.data import DataLoader, DistributedSampler
from transformers import EvaluationStrategy, PreTrainedModel, set_seed, TrainerState, WEIGHTS_NAME, is_wandb_available

if is_wandb_available():
    from transformers.integrations import WandbCallback
    from src.visualization.wandb_callbacks import MyWandbCallback

from transformers.trainer import Trainer, is_torch_tpu_available
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import PredictionOutput, TrainOutput

from src.data.make_dataset_document_electra import make_datasets_document_electra
from src.data.utils import get_dataset_info
from src.features.features_document_electra import DataCollatorForDocumentElectra
from src.models.metrics import ComputeMetricsDocumentElectraForPretraining
from src.models.modeling_document_electra import DocumentElectraConfig, DocumentElectraPretrainingModel, \
    DocumentElectraPretrainingModelOutput
from src.models.utils import get_tokenizer, MyTrainingArguments, my_set_seed
from src.visualization.tensorboard_utils import get_tensorboard_experiment_id, MySummaryWriter, to_sanitized_dict

if is_torch_tpu_available():
    # noinspection PyUnresolvedReferences
    import torch_xla.core.xla_model as xm
    # noinspection PyUnresolvedReferences
    import torch_xla.debug.metrics as met
    # noinspection PyUnresolvedReferences
    import torch_xla.distributed.parallel_loader as pl

import sys
IN_COLAB = 'google.colab' in sys.modules

logger = logging.getLogger(__name__)

_default_settings = {}

if torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory < 2 ** 30 * 6):
    # Less 6 GB GPU memory
    _default_settings["max_length"] = 128 * 16  # 2048
    _default_settings["max_position_embeddings"] = 4096
    _default_settings["per_device_train_batch_size"] = 1
    _default_settings["per_device_eval_batch_size"] = 1
    _default_settings["num_hidden_layers"] = 12
    _default_settings["effective_batch_size"] = 4
    _default_settings["lr"] = 5e-4
else:
    _default_settings["max_length"] = 128 * 16  # 2048
    _default_settings["max_position_embeddings"] = 4096
    _default_settings["per_device_train_batch_size"] = 1
    _default_settings["per_device_eval_batch_size"] = 1
    _default_settings["num_hidden_layers"] = 12
    _default_settings["effective_batch_size"] = 4
    _default_settings["lr"] = 5e-4


# Hack to prevent tokenizer to fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# noinspection PyTypeChecker
def train_model_pretraining(
        dataset_name: Union[str, List[str]] = None,
        dataset_sampling: Union[float, List[float]] = None,
        dataset_dir: str = None,
        cache_dir: str = None,
        output_dir: str = "checkpoints",
        tokenizer_path: str = r"models/ByteLevelBPETokenizer-vocab_size=30522-min_frequency=2",
        gradient_accumulation_steps: int = (_default_settings["effective_batch_size"] /
                                            _default_settings["per_device_train_batch_size"]),
        # 1 for ElectraSmall but aggregation across docs decrease variance

        per_device_train_batch_size: int = _default_settings["per_device_train_batch_size"],
        per_device_eval_batch_size: int = _default_settings["per_device_eval_batch_size"],

        lr: int = _default_settings["lr"],  # 5e-4 for ElectraSmall * gradient_accumulation_steps
        max_grad_norm: float = 1,  # 1 for Electra
        # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py
        embedding_size: int = 128,  # Output and input embedding size, 128 for ElectraSmall
        hidden_size: int = 256,  # 256 for ElectraSmall
        chunk_length: int = 128,
        num_hidden_layers: int = _default_settings["num_hidden_layers"],  # 12 for Electra Small
        num_attention_heads: int = 4,
        layer_depth_offset: int = -1,

        max_sentence_length: int = 128,
        max_sentences: int = 128,
        max_length: int = _default_settings["max_length"],
        max_position_embeddings: int = _default_settings["max_position_embeddings"],

        intermediate_size: int = 1024,  # 1024 for ElectraSmall
        hidden_act: str = "gelu",

        generator_size: float = 0.25,  # Generator size multiplier
        generator_layer_size: float = 1.0,  # Generator layer size multiplier
        mlm_probability: float = 0.15,
        mlm_replacement_probability: float = 0.85,
        temperature: float = 1.,  # 1 for ElectraSmall
        discriminant_loss_factor: float = 50,  # Factor for the discriminant loss, Default 50 in Electra (lambda)

        hidden_dropout_prob: float = 0.1,  # 0.1 for ElectraSmall
        attention_probs_dropout_prob: float = 0.1,  # 0.1 for ElectraSmall
        logging_steps: int = _default_settings["effective_batch_size"] * 3,  # Default logging every 3 grad updates
        eval_steps: int = _default_settings["effective_batch_size"] * 3 * 10,
        save_steps: int = _default_settings["effective_batch_size"] * 800,  # Every 800 steps (about every hour)
        # Default evaluation every 10 logging steps (30 grad updates)
        seed: int = 42,
        training_set_random_seed: int = 42,
        valid_test_split_random_seed: int = 42,
        training_set_size: int = -1,
        num_proc: int = 0,
        gradient_checkpointing: bool = False,  # Decrease mem by 5/10% but increase compute cost
        pretrain_path: str = None,
        warmup_steps: int = int(10000),  # 10000 for ElectraSmall
        weight_decay: float = 0.01,  # 0.01 for ElectraSmall
        tensorboard_tracking_folder: str = "tensorboard",

        metric_to_train_generator: str = "eval_is_fake_token_AUC",
        threshold_to_train_generator: float = 0.0,
        relative_position_embeddings: bool = False,
        sequence_embeddings: bool = False,
        experiment_name: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Main function to launch the pretraining task
    :param experiment_name: Name of experiment
    :param dataset_name: name or list of names of datasets from HuggingFace NLP library.
    Default [Wikipedia-20200501.en', 'bookcorpus']
    :param dataset_sampling: ratio for downsampling
    Default [1.0, 0.3]
    :param dataset_dir: Directory path for the cache for the HuggingFace datasets library
    :param cache_dir: Directory to store the cache for the processed datasets.
    :param output_dir: Checkpoints folder
    :param tokenizer_path: Path to tokenizer
    :param gradient_accumulation_steps: Default 32.
    :param per_device_train_batch_size: Batch size per GPU. Default 1.
    :param per_device_eval_batch_size: Batch size per GPU. Default 2.
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
    :param logging_steps: Logging steps (show train loss) in number of samples (and not in gradient updates)
    :param eval_steps: Evaluate steps (evaluate and show val loss) in number of samples (and not in gradient updates)
        If a float is provided, then the evaluation step is every eval_steps * steps per epoch.
    :param save_steps: Number of steps to perform a checkpoint. Default every 800 steps (equivalent every hour)

    :param weight_decay: Default 0.01 for ElectraSmall
    :param warmup_steps: Default 10000 samples like ElectraSmall
    :param num_proc: Number of processor for data preprocessing steps
    :param discriminant_loss_factor: factor for the discriminant loss. Default 50 as Electra
    :param generator_size: factor to decrease generator complexity compared to the discriminant. Default 25%
    :param generator_layer_size:  factor to decrease generator complexity
    compared to the discriminant (number of layers). Default 1.0
    compared to the discriminant at token level (only for num_layers). Default 25%
    :param mlm_replacement_probability: Probability of replacement for selected tokens. Default: 0.85
    :param mlm_probability: Probability to corrupt tokens. Default: 0.15
    :param temperature: Temperature for the MLM sampling. Default 1 like Electra
    :param training_set_size: Default -1 to use all possible training set.
    :param seed: Seed used in everything except for dataset split and shuffling
    :param training_set_random_seed: Seed used only for shuffle the training set
    :param valid_test_split_random_seed: Seed used only for the split between test and validation sets.
        Required to ensure the validation set remains the same if seed is used.
    :param attention_probs_dropout_prob
    :param tensorboard_tracking_folder:
    :param pretrain_path: Path to the pretrained checkpoints. Model weights and optimizer states will be loaded.
            This will allow to continue training.
    :param gradient_checkpointing: Default False
    :param metric_to_train_generator: Metric to monitor for deciding to train the generator or not
    :param threshold_to_train_generator: Threshold to switch on the generator training (training if above). Default: 0.0
    :param relative_position_embeddings: Use relative position embeddings. Default True
    :param sequence_embeddings: Use sequence embeddings (number of sentences). Default True
    :return:
    """
    my_set_seed(seed=seed)

    if dataset_name is None:
        dataset_name = ["openwebtext"]
    if dataset_sampling is None:
        dataset_sampling = [1.0]

    if experiment_name is None:
        experiment_name = "pretraining"

    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)

    if isinstance(dataset_name, List):
        dataset_name = [name.split("-") for name in dataset_name]
        dataset_name = [(name[0], name[1] if len(name) > 1 else None) for name in dataset_name]
        dataset_infos = [get_dataset_info(dataset_name=name, dataset_subset=subset) for name, subset in dataset_name]
    else:
        dataset_name = dataset_name.split("-")
        dataset_name, dataset_subset = (dataset_name[0], dataset_name[1] if len(dataset_name) > 1 else None)
        dataset_infos = [get_dataset_info(dataset_name=dataset_name, dataset_subset=dataset_subset)]

    config = DocumentElectraConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        layer_depth_offset=layer_depth_offset,
        max_sentence_length=max_sentence_length,
        max_sentences=max_sentences,
        max_position_embeddings=max_position_embeddings,
        max_length=max_length,
        gradient_checkpointing=gradient_checkpointing,
        generator_size=generator_size,
        generator_layer_size=generator_layer_size,
        mlm_probability=mlm_probability,
        mlm_replacement_probability=mlm_replacement_probability,
        temperature=temperature,
        discriminant_loss_factor=discriminant_loss_factor,
        chunk_length=chunk_length,
        relative_position_embeddings=relative_position_embeddings,
        sequence_embeddings=sequence_embeddings,
    )
    model = DocumentElectraPretrainingModel(config=config)
    # Torch Script is not compatible with Transformers model
    # See https://github.com/huggingface/transformers/pull/6846
    # traced_model = torch.jit.script(model)

    train_set, val_set, test_set = make_datasets_document_electra(
        tokenizer=tokenizer,
        dataset_info=dataset_infos,
        cache_dir=cache_dir,
        dataset_dir=dataset_dir,
        dataset_sampling=dataset_sampling,
        training_set_size=training_set_size,

        # 100 steps for evaluation in ElectraSmall
        # We use 200 documents instead of 1 for ElectraSmall to have a better estimate (no random shuffle in our case)
        # https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/configure_pretraining.py#L50
        validation_set_size=100 * per_device_eval_batch_size,
        test_set_size=0,

        training_set_random_seed=training_set_random_seed,
        valid_test_split_random_seed=valid_test_split_random_seed,
        num_proc=1)  # Multiple processors will need a dataset copy (so it increase the disk space requirement)

    train_data_collator = DataCollatorForDocumentElectra(config=config)

    effective_train_batch_size = per_device_train_batch_size * gradient_accumulation_steps

    training_args = MyTrainingArguments(
        output_dir=get_tensorboard_experiment_id(experiment_name=experiment_name,
                                                 tensorboard_tracking_folder=output_dir),
        overwrite_output_dir=True,
        #        num_train_epochs=epochs,
        # equivalent to 1M (original paper use 20M with 128 batches instead of 6M in this preprocessing)
        # This gives an equivalent of 6.4 epochs for these 1M steps
        max_steps=1e6,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_steps=int(save_steps / effective_train_batch_size),
        save_total_limit=200,
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
        optimizer="AdamW",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,  # If lower, this may create an issue for amp.
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        dataloader_num_workers=num_proc,
        training_set_size=training_set_size,
        training_set_random_seed=training_set_random_seed,
        valid_test_split_random_seed=valid_test_split_random_seed,

        metric_to_train_generator=metric_to_train_generator,
        threshold_to_train_generator=threshold_to_train_generator
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

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=train_data_collator,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=ComputeMetricsDocumentElectraForPretraining(tb_writer=tb_writer,
                                                                    hparams={**to_sanitized_dict(config),
                                                                             **to_sanitized_dict(training_args),
                                                                             },
                                                                    sentence_predictions=False),
        mask_token_id=config.mask_token_id,
        tb_writer=tb_writer
    )

    if is_wandb_available():
        # Workaround to force the creation of a new Wandb callback for each run if we launches several run
        trainer.pop_callback(WandbCallback)
        trainer.add_callback(MyWandbCallback)

    if pretrain_path is not None:
        pretrain_path = Path(pretrain_path)
        assert pretrain_path.exists() and pretrain_path.is_dir()

        logger.info(f"Load pretrained weights from {pretrain_path}")
        pretrain_dict = torch.load(str(pretrain_path / "pytorch_model.bin"))
        model.load_state_dict(state_dict=pretrain_dict)  # Default is strict=True
        logger.info(f"Load pretrained weights from {pretrain_path} Completed")

        trainer.train(model_path=str(pretrain_path))

    else:
        trainer.train()

    eval_output: Dict[str, float] = trainer.evaluate()
    logger.info(eval_output)

    return eval_output


class MyTrainer(Trainer):
    """
    Custom method to include:
    - Logging of several losses
    - Management of variable size of outputs (binary classification and token level and sentence level)
    - Monitoring of one metric to decide to train the generator or not
    """

    args: MyTrainingArguments

    def __init__(
            self,
            compute_metrics: ComputeMetricsDocumentElectraForPretraining,
            mask_token_id: int,
            model: DocumentElectraPretrainingModel,
            **kwargs,
    ):
        super(MyTrainer, self).__init__(model=model, **kwargs)
        assert isinstance(model, DocumentElectraPretrainingModel)
        if self.args.fp16:
            assert version.parse(torch.__version__) >= version.parse("1.6"), "No AMP management with APEX"

        self._logging_loss_scalar = 0
        self.compute_metrics: ComputeMetricsDocumentElectraForPretraining = compute_metrics
        self.mask_token_id = mask_token_id

    """

    training_step should return more losses

    compute_loss also

    Train should use the new type from Training step


    """

    def training_step(self, model: nn.Module,
                      inputs: Dict[str, Union[torch.Tensor, Any]]) -> DocumentElectraPretrainingModelOutput:
        """
        Perform a training step on a batch of inputs.

        Extra logic: keep track of more than one loss

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16:
            with autocast():
                output: DocumentElectraPretrainingModelOutput = self.compute_loss(model, inputs)
        else:
            output: DocumentElectraPretrainingModelOutput = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            output.loss /= (self.args.gradient_accumulation_steps * self.args.n_gpu)
            output.generator_loss /= (self.args.gradient_accumulation_steps * self.args.n_gpu)
            output.discriminant_loss /= (self.args.gradient_accumulation_steps * self.args.n_gpu)
            output.discriminant_token_loss /= (self.args.gradient_accumulation_steps * self.args.n_gpu)

        if self.args.fp16:
            self.scaler.scale(output.loss).backward()
        else:
            output.loss.backward()

        output.loss = output.loss.detach()
        return output

    def prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model: DocumentElectraPretrainingModel = self.model
        assert isinstance(model, DocumentElectraPretrainingModel)

        # multi-gpu eval
        if self.args.n_gpu > 1:
            # noinspection PyTypeChecker
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        # Custom logic
        losses_host: torch.Tensor = torch.tensor(0.0, device=model.device)

        # Generator
        generator_losses_host: torch.Tensor = torch.tensor(0.0, device=model.device)
        mlm_preds_host: List[torch.Tensor] = []
        sampled_mlm_preds_host: List[torch.Tensor] = []
        generator_labels_host: List[torch.Tensor] = []

        # Discriminant
        discriminant_losses_host: torch.Tensor = torch.tensor(0.0, device=model.device)

        discriminant_token_losses_host: torch.Tensor = torch.tensor(0.0, device=model.device)
        is_fake_preds_host: List[torch.Tensor] = []
        is_fake_labels_host: List[torch.Tensor] = []

        # End Custom logic

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):

            # Custom logic
            output = self.prediction_step(model, inputs, prediction_loss_only)
            assert output.loss is not None

            losses_host += output.loss.detach()

            # Generator
            generator_losses_host += output.generator_loss

            # Only corrupted tokens
            mask_tokens_mask = output.labels_generator.reshape(-1).ne(-100)

            mlm_preds_host += [output.mlm_input_ids.reshape(-1)[mask_tokens_mask]]
            sampled_mlm_preds_host += [output.sampled_input_ids.reshape(-1)[mask_tokens_mask]]
            generator_labels_host += [output.labels_generator.reshape(-1)[mask_tokens_mask]]

            # Discriminant
            discriminant_losses_host += output.discriminant_loss

            discriminant_token_losses_host += output.discriminant_token_loss

            non_special_tokens = output.labels_at_token_level.reshape(-1).ne(-100)
            is_fake_preds_host += [output.is_fake_logits.reshape(-1)[non_special_tokens]]
            is_fake_labels_host += [output.labels_at_token_level.reshape(-1)[non_special_tokens]]

            # End Custom Logic

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        # Custom logic
        metrics = self.compute_metrics(predictions=[torch.cat(is_fake_preds_host).cpu(),
                                                    torch.cat(mlm_preds_host).cpu(),
                                                    torch.cat(sampled_mlm_preds_host).cpu()],
                                       labels=[torch.cat(is_fake_labels_host).cpu(),
                                               torch.cat(generator_labels_host).cpu(),
                                               torch.cat(generator_labels_host).cpu()])

        metrics["loss"] = (losses_host / num_examples).cpu().item()
        metrics["generator_loss"] = (generator_losses_host / num_examples).cpu().item()
        metrics["discriminant_loss"] = (discriminant_losses_host / num_examples).cpu().item()
        metrics["discriminant_token_loss"] = (discriminant_token_losses_host / num_examples).cpu().item()

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        if model.train_generator and (self.args.metric_to_train_generator in metrics) and (
                metrics[self.args.metric_to_train_generator] <= self.args.threshold_to_train_generator):
            model.train_generator = False
        if not model.train_generator and (self.args.metric_to_train_generator in metrics) and (
                metrics[self.args.metric_to_train_generator] > self.args.threshold_to_train_generator):
            model.train_generator = True
        metrics[f"train_generator"] = float(model.train_generator)

        # Metrics contain already all information. No need to add predictions and labels.
        # Anyway, not possible in this case since the dimension differs between labels
        return PredictionOutput(predictions=np.array([]), label_ids=np.array([]), metrics=metrics)

        # End Custom Logic

    # noinspection PyUnresolvedReferences
    def train(self, model_path: Optional[str] = None,
              trial: Dict[str, Any] = None):  # Comment optuna library to prevent dependencies
        # trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)

            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer: Optional[Optimizer] = None
            # noinspection PyUnresolvedReferences,PyProtectedMember
            self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = int(self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                ))
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = int(math.ceil(self.args.num_train_epochs))
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))
            reissue_pt_warnings(caught_warnings)

        # Mixed precision training with apex (torch < 1.6)
        model = self.model

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
        # find_unused_parameters breaks checkpointing as per
        # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            # noinspection PyUnresolvedReferences
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", self.state.global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(self.args.device)

        # Custom logic
        logs = {}
        tr_generator_loss, tr_discriminant_loss = 0.0, 0.0
        tr_discriminant_token_loss, tr_discriminant_sentence_loss = 0.0, 0.0
        # End Custom Logic

        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # Custom Logic
                if (
                        ((step + 1) % self.args.gradient_accumulation_steps != 0)
                        and self.args.local_rank != -1
                ):
                    with model.no_sync():
                        output = self.training_step(model, inputs)
                else:
                    output = self.training_step(model, inputs)

                tr_loss += output.loss
                tr_generator_loss += output.generator_loss
                tr_discriminant_loss += output.discriminant_loss
                tr_discriminant_token_loss += output.discriminant_token_loss
                # End Custom Logic

                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        self.args.gradient_accumulation_steps >= steps_in_epoch == (step + 1)
                ):
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    # Custom Logic
                    logs["generator_loss"] = tr_generator_loss
                    logs["discriminant_loss"] = tr_discriminant_loss
                    logs["discriminant_token_loss"] = tr_discriminant_token_loss

                    self._maybe_log_save_evalute(tr_loss, model, trial, epoch, logs=logs)
                    logs = {}
                    tr_generator_loss, tr_discriminant_loss = 0.0, 0.0
                    tr_discriminant_token_loss = 0.0
                    # End Custom Logic

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)

            # Custom Logic
            logs["generator_loss"] = tr_generator_loss
            logs["discriminant_loss"] = tr_discriminant_loss
            logs["discriminant_token_loss"] = tr_discriminant_token_loss
            self._maybe_log_save_evalute(tr_loss, model, trial, epoch, logs=logs)
            # End Custom Logic

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(model, PreTrainedModel):
                self.model = model.from_pretrained(self.state.best_model_checkpoint)
                self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step)

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> DocumentElectraPretrainingModelOutput:
        """
            Extra logic: add the parameter to return all outputs and also track all losses (not just the combined one)

        :param model:
        :param inputs:
        :param prediction_loss_only:
        :return:
        """
        inputs = self._prepare_inputs(inputs)
        inputs["loss_only"] = False

        with torch.no_grad():
            return model(**inputs)

    def compute_loss(self, model, inputs) -> DocumentElectraPretrainingModelOutput:
        """
            Extra logic: return all combined loss, generator loss and discriminator loss.
            The default HuggingFace implementation is only for one loss to monitor.

        :param model:
        :param inputs:
        :return:return all combined loss, generator loss and discriminator loss.
        """
        inputs["loss_only"] = True
        outputs: DocumentElectraPretrainingModelOutput = model(**inputs)

        return outputs

    # noinspection SpellCheckingInspection
    def _maybe_log_save_evalute(self, tr_loss, model, trial, epoch, logs=None):
        """
            Custom logic is to already provide a logs dictionary with the additional losses
        :param tr_loss:
        :param model:
        :param trial:
        :param epoch:
        :param logs:
        :return:
        """
        if logs is None:
            logs = {}
        if self.control.should_log:
            tr_loss_scalar = tr_loss.item()
            logs["loss"] = (tr_loss_scalar - self._logging_loss_scalar) / self.args.logging_steps
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )
            self._logging_loss_scalar = tr_loss_scalar

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
