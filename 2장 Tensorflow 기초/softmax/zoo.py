import tensorflow as tf
import numpy as np

# Load dataset
zoo = np.loadtxt("data-04-zoo.csv", delimiter=',', dtype=np.float32)

# number of classes
NUMBER_OF_CLASSES = 7

# define x, y
x_data = zoo[:, 0:-1]
print("x_data.shape: ", x_data.shape)
y_data = zoo[:, [-1]]
print("y_data.shape: ", y_data.shape)

Y = tf.placeholder(tf.int32, [None, 1])  # 0~6, shape=(?,1)
Y_onehot = tf.one_hot(Y, NUMBER_OF_CLASSES)  # one hot shape=(?,1,7)
Y_onehot = tf.reshape(Y_onehot, [-1, NUMBER_OF_CLASSES])  # shape=(?,7)
