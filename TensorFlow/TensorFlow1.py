# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

greeting = tf.constant('Hello Google Tensorflow')
sess = tf.Session()
result = sess.run(greeting)
print(result)
sess.close()