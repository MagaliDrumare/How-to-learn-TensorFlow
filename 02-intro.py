#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:36:16 2017

@author: magalidrumare
"""



"""TensorFlow separates the definition of computations from their execution even further 
by having them happen in separate places: a graph defines the operations, 
but the operations only happen within a session. 
Graphs and sessions are created independently. 
A graph is like a blueprint, and a session is like a construction site"""

import tensorflow as tf 

input_value = tf.constant(1.0)
weight = tf.Variable(0.8)
output_value = weight * input_value
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
sess.run(output_value)
 
import tensorflow as tf 
x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
sess.run(y)
#summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
file_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)
# à taper dans la console : tensorboard --logdir=log_simple_graph
#ouvrir le navigateur web :  http://0.0.0.0:6006


import tensorflow as tf

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')
y_ = tf.constant(0.0, name='correct_value')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for value in [x, w, y, y_, loss]:
  tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()

sess = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(100):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)
    print(sess.run(loss))
# à taper dans la console : tensorboard --logdir=log_simple_stats
#ouvrir le navigateur web :  http://0.0.0.0:6006

