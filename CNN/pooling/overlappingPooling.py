import tensorflow as tf

x = tf.constant([0, 0, 5, 2, 4, 2, 7, 2, 8])
x = tf.reshape(x, [1, 9, 1])
# if stride size = kernel_window_size => traditional pooling
max_pool_1d = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, padding='valid')

print(max_pool_1d(x))