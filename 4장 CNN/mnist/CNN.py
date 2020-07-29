import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

graph = tf.get_default_graph()

# read dataset
mnist_dataset = input_data.read_data_sets("./", one_hot=True)

# set parameters
epochs = 20
batch_size = 100

# define variables
# reshape input to a 2D image
X_raw = tf.placeholder(tf.float32, [None, 784])
X = tf.reshape(X_raw, [-1, 28, 28, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")

# convolution layer 1
Kernel1 = tf.Variable(tf.random_normal(
    [3, 3, 1, 32], stddev=0.1), name="kernel1")
# convolution output: 28X28X32
Conv1 = tf.nn.conv2d(X, Kernel1, strides=[
                     1, 1, 1, 1], padding="SAME", name="conv1")
# activation ReLu
Conv1 = tf.nn.relu(Conv1, name="relu1")
# max pooling output: 14X14X32
Conv1 = tf.nn.max_pool(Conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="SAME", name="max_pool1")

# convolution layer2
Kernel2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1), name="kernel2")
# convolution output: 14X14X64
Conv2 = tf.nn.conv2d(Conv1, Kernel2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
# activation ReLu
Conv2 = tf.nn.relu(Conv2, name="relu2")
# max pooling output: 7X7X64
Conv2 = tf.nn.max_pool(Conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="SAME", name="max_pool2")

# flatten to a 1D vector
Conv2 = tf.reshape(Conv2, [-1, 7*7*64], name="flatten")

# fully connected layer
# tf.get_variable is used for variable reuse
# if a variable does not exists, creates a new variable
W = tf.get_variable(
    "W", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name='b',
                    shape=[10], initializer=tf.random_normal_initializer())
hypothesis = tf.matmul(Conv2, W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with graph.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            avg_loss = 0
            total_batch = int(mnist_dataset.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_x, batch_y = mnist_dataset.train.next_batch(batch_size)
                feed_dict = {X_raw: batch_x, Y: batch_y}
                l, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
                avg_loss += l/total_batch
            print("Epoch: {}, loss: {}".format(epoch+1, avg_loss))

        correct_prediction = tf.equal(
            tf.argmax(hypothesis, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy: ", sess.run(accuracy, feed_dict={X_raw: mnist_dataset.test.images,
                                                          Y: mnist_dataset.test.labels}))

        test_index = random.randint(0, mnist_dataset.test.num_examples - 1)

        print("Label: ", sess.run(
            tf.argmax(mnist_dataset.test.labels[test_index:test_index+1], axis=1)))
        print("Prediction: ", sess.run(tf.argmax(hypothesis, axis=1),
                                    feed_dict={X_raw: mnist_dataset.test.images[test_index:test_index+1]}))

        plt.imshow(mnist_dataset.test.images[test_index:test_index + 1].reshape(28, 28),
                cmap="Greys", interpolation="nearest")
        plt.show()
