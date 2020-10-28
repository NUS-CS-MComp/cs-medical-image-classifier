import tensorflow as tf

from utils.logging import logger


def assign_gpu(memory_limit: int):
    """
    Helper function to assign GPU units for training
    :return: None
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit
                    )
                ],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            logger.info(
                f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs"
            )
        except RuntimeError as e:
            logger.error(e)
