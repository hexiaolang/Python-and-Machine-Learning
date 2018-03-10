# -*- coding:utf-8 -*-
import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])  # 1*2
matrix2 = tf.constant([[2.], [2.]])  # 2*1
product = tf.matmul(matrix1, matrix2)
linear = tf.add(product,  tf.constant(2.0))

with tf.Session() as sess:
    result = sess.run(linear)
    print(result)
