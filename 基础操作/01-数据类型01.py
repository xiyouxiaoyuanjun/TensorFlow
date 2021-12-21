import tensorflow as tf
import numpy as np
#creat tensor

tensor1=tf.constant(1)
print(tensor1)

tensor2=tf.constant(1.)
print(tensor2)

tensor3=tf.constant(2.,dtype=tf.double)
print(tensor3)

tensor4=tf.constant([True,False])
print(tensor4)

tensor5=tf.constant('hello world.')
print(tensor5)

with tf.device("cpu"):
    a=tf.constant([1])
with tf.device("gpu"):
    b=tf.constant(4)

print(a.device)
print(b.device)

aa=b.numpy()
print(aa)

