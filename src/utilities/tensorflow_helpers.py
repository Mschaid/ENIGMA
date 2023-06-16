

import os
import datetime
import tensorflow as tf


def set_tensorboard(model_id, save_directory):
    """
    Set up TensorBoard for model training. Returns the TensorBoard callback.

    Parameters
    ----------
    save_directory : str
        The directory to save the TensorBoard logs.
    model_id : str
        The ID of the model.

    Returns
    -------
    tensorflow.keras.callbacks.TensorBoard
        The TensorBoard callback.
    """
    
    if not os.path.exists(save_directory): os.makedirs(save_directory)
    
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    logs_dir = f"{save_directory}/{model_id}/{date_time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
    return tensorboard_callback