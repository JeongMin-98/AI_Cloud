# status initialize

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'



# out of memory

import tensorflow as tf

with tf.Graph().as_default():

  gpu_options = tf.GPUOptions(allow_growth=True)