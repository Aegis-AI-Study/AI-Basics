import tensorflow as tf
from matplotlib import pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define input and output
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# define weights and bias
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

# define parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 50
num_iterations = int(mnist.train.num_examples / batch_size)

output = tf.matmul(X, W) + b

# define cost/loss & optimizer
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=output, labels=Y
    )
)

# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.arg_max(output, dimension=1), tf.arg_max(Y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_loss = 0

        for i in range(num_iterations):
            batch_x, batch_y = mnist.train.next_batch(batch_size)  # call next batch
            l, _ = sess.run([loss, optimizer], feed_dict={X: batch_x, Y:batch_y})
            avg_loss += l / num_iterations

        print("Epoch: ", '{}'.format(epoch+1, "04d"), "Cost: ", '{}'.format(avg_loss, '.5f'))

    print("Learning finished")

    test_index = random.randint(0, mnist.test.num_examples - 1)

    print("Label: ", sess.run(tf.argmax(mnist.test.labels[test_index:test_index+1], axis=1)))
    print("Prediction: ", sess.run(tf.argmax(output, axis=1),
                                   feed_dict={X: mnist.test.images[test_index:test_index+1]}))

    plt.imshow(mnist.test.images[test_index:test_index+1].reshape(28,28), cmap="Greys", interpolation="nearest")
    plt.show()
