import tensorflow as tf
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if __name__ == '__main__':
    a = layers.Dense(4)
    a.build(input_shape=[None, 2])
    x = tf.constant([[1, 2], [3, 4]])
    b = a(x)
    print(b)
