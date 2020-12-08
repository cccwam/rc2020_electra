import wandb
from transformers import is_torch_tpu_available
from transformers.integrations import WandbCallback
import os

from transformers.utils import logging

logger = logging.get_logger(__name__)

class MyWandbCallback(WandbCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases
    <https://www.wandb.com/>`__.
    """

    def setup(self, args, state, model):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        `here <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely.
        """
#        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}
            if getattr(model, "config", None) is not None:
                combined_dict = {**model.config.to_dict(), **combined_dict}
            wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=combined_dict, name=args.run_name,
                       reinit=True)  # CUSTOM LOGIC TO REINIT WANDB AFTER EACH RUN
            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                wandb.watch(model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """

        :param args:
        :param state:
        :param control:
        :param model:
        :param kwargs:
        """
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """

        :param args:
        :param state:
        :param control:
        :param model:
        :param logs:
        :param kwargs:
        """
#        if not self._initialized:
#            self.setup(args, state, model)
        if state.is_world_process_zero:
            wandb.log(logs, step=state.global_step)