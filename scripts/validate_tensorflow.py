import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if GPUs are detectable
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print("GPUs are detectable.")
    for gpu in gpus:
        print("Available GPU:", gpu)
else:
    print("No GPUs are detectable.")

print("TensorFlow is using GPU:", tf.test.is_built_with_cuda() and tf.test.is_gpu_available())
