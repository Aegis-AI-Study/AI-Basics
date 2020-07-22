#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv("./xor.csv")

x1 = data['x1']
x2 = data['x2']
x_data = np.array([[x_1, x_2] for x_1, x_2 in zip(x1.values, x2.values)], dtype=np.float32)
y_data = data['y']

# define tf variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# define our weight matrix
W = tf.Variable(tf.random_uniform([2, 1]), name="weight")
# define our bias vector
b = tf.Variable(tf.random_uniform([1]), name="bias")

# define our hypothesis
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# define cost(loss) function
# cost : binary cross entropy loss
cost = tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# define optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# compute accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# launch graph
with tf.Session() as sess:
    # initialize variable
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nhypothesis: ", h, "\nPredicted: ", c, "\nAccuracy: ", a)