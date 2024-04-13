import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.test.is_gpu_available():
    print("TensorFlow is able to use the GPU!")
else:
    print("TensorFlow cannot use the GPU.")
