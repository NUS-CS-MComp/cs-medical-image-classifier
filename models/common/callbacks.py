import pathlib
from datetime import datetime
from typing import Dict

import tensorflow as tf

from utils.configuration import MODEL_CHECKPOINT_PATH


def generate_model_checkpoint_callback(
    model_script_path: str,
    is_fine_tuning: bool = False,
    tag: str = "",
    state: Dict = None,
):
    """
    Helper function to generate checkpoint callback
    :param model_script_path: model script path passed from __file__
    :param is_fine_tuning: boolean flag for fine tuning
    :param tag: extra tag to append to model name
    :param state: extra state parameters to be appended to folder name
    :return: callback object
    """

    def format_state_string(state_dict: Dict):
        if state_dict is None:
            return ""
        else:
            return "_" + "_".join(
                [str(value) for key, value in state_dict.items()]
            )

    model_name = pathlib.Path(model_script_path).stem
    if tag != "":
        model_name = model_name + f"_{tag}"
    model_timestamp = datetime.now().strftime("%m%d%H%M")
    model_parent_path = (
        MODEL_CHECKPOINT_PATH
        / f"{model_name}/{model_timestamp}{format_state_string(state)}"
    )
    fine_tuning_flag = "fine-tuned-" if is_fine_tuning else ""
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_parent_path.absolute()}/{fine_tuning_flag}epoch-{{epoch:02d}}-loss-{{val_loss:.4f}}",
        verbose=1,
        save_weights_only=True,
        monitor="val_loss",
        save_best_only=True,
    )


def generate_early_stopping_callback():
    """
    Helper function to generate early stopping callback
    :return: callback object
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, min_delta=0.001, mode="min"
    )


def generate_learning_rate_schedule_callback(
    start_learning_rate: float = 0.001,
    ramp_up_epochs: int = 10,
    exp_decay=0.01,
):
    """
    Helper function to generate learning rate schedule callback
    :param start_learning_rate: initial learning rate
    :param ramp_up_epochs: epoch to start decaying
    :param exp_decay: exponential decay rate
    :return: callback object
    """

    def schedule(epoch):
        """
        Return exponential decay learning rate schedule
        :param epoch: current epoch
        :return: learning rate schedule thunk
        """

        def learning_rate(curr_epoch, lr, epoch_threshold, decay):
            return (
                lr
                if curr_epoch < epoch_threshold
                else lr * tf.math.exp(-decay * epoch)
            )

        return learning_rate(
            epoch, start_learning_rate, ramp_up_epochs, exp_decay
        )

    return tf.keras.callbacks.LearningRateScheduler(
        schedule=schedule, verbose=1
    )
