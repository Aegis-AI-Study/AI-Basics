#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv("./xor.csv")

x1 = data["x1"]
x2 = data["x2"]
x_data = np.array([[_1, _2] for _1, _2 in zip(x1.values, x2.values)], dtype=np.float32)
y_data = data['y']

# define placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# define variables
W1 = tf.Variable(tf.random_normal([2, 2]), name="weight1")
b1 = tf.Variable(tf.random_normal([2]), name="bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")
output = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# define loss function
loss = -tf.reduce_mean(Y * tf.log(output) + (1 - Y) * tf.log(1 - output))  # binary cross entropy loss

# define model
model = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# compute accuracy
predicted = tf.cast(output > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# launch graph
with tf.Session() as sess:
    # initialize tf variables
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(model, feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, sess.run(loss, feed_dict={X: x_data, Y: y_data}), sess.run(W1, W2))

    # print result
    o, p, a = sess.run([output, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\noutput: ", o, "\npredicted: ", p, "\naccuracy: ", a)
