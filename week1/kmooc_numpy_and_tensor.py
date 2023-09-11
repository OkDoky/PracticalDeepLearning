import numpy as np
import tensorflow as tf

a = tf.constant([[1,2],[3,4]])
b = tf.add(a,1)

a.numpy()
print(a)
b.numpy()
print(b)