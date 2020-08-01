import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import PIL.Image as pilimg
import PIL

graph = tf.get_default_graph()

# convolution layer
def Convnet(input_tensor, input_channels, output_channels, name,
                stddev=0.1, padding="SAME",
                kernel_shape=(3,3), conv_stride=[1,1,1,1]):

        # set kernel
        kernel = tf.Variable(tf.random_normal(
            shape=(*kernel_shape, input_channels, output_channels),
            stddev=stddev,
            name=name+"_kernel"
        ))

        # convolution
        Conv1 = tf.nn.conv2d(
            input=input_tensor,
            filter=kernel,
            strides=conv_stride,
            padding=padding,
            name=name
        )
        return Conv1

# fully connected layer
def FCnet(input_tensor, output_channels, name):
    '''
    input tensor shape : (1, ?)
    '''
    W = tf.get_variable(name+"/W", shape=(input_tensor.shape.as_list()[1], output_channels),
                        dtype=tf.float32)
    b = tf.get_variable(name+"/b", shape=(output_channels),
                        dtype=tf.float32)

    output = tf.matmul(input_tensor, W) + b
    return output

class VGG16():
    def __init__(self, input_image_vector: list, class_labels: list, epochs: int=20, batch_size: int=100):
        '''
        input_image: PIL Image
        '''
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_image_vector = input_image_vector
        self.class_labels = class_labels
        self.total_batch = int(len(input_image_vector)/batch_size)

    # returns a list of images
    def batch_generator(self):
        for i in range(0, self.total_batch, self.batch_size):
            image_batch = self.input_image_vector[i:i+self.batch_size]
            class_label_batch = self.class_labels[i:i+self.batch_size]
            yield image_batch, class_label_batch

    @classmethod
    def train(self, num_classes):

        conv_strides = (1,1,1,1)
        pooling_strides = (1,2,2,1)
        padding = "SAME"
        
        X_raw = tf.placeholder(dtype=tf.float32, shape=(224,224,3), name="X_raw")
        X = tf.reshape(X_raw, (-1, 224, 224, 3), name="X")
        Y = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name="Y")

        # block 1
        # input: 224, 224, 3
        # output: 112, 112, 64
        kernel1_1 = tf.get_variable(name="kernel1_1", shape=(3,3,3,64),
                initializer=tf.random_normal_initializer(stddev=0.1))
        conv1_1 = tf.nn.conv2d(X, kernel1, strides=conv_strides,
                        padding=padding, name="conv1_1")
        conv1_1 = tf.nn.relu(conv1_1)

        kernel1_2 = tf.get_variable(name="kernel1_2", shape=(3,3,64,64),
                initializer=tf.random_normal_initializer(stddev=0.1))
        conv1_2 = tf.nn.conv2d(conv1_1, kernel1_2, strides=conv_strides,
                        padding=padding, name="conv1_2")
        conv1_2 = tf.nn.relu(conv1_2)

        pool1 = tf.nn.max_pool(conv1_2, ksize=pooling_strides, strides=pooling_strides,
        padding=padding, name="pool1")

        # block 2
        # input: 112, 112, 64
        # output: 56, 56, 128
        kernel2_1 = tf.get_variable(
            name="kernel2_1", shape=(3,3,64,128),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv2_1 = tf.nn.conv2d(
            pool1, kernel2_1,
            strides=conv_strides, padding=padding, name="conv2_1"
        )
        conv2_1 = tf.nn.relu(conv2_1)
        
        kernel2_2 = tf.get_variable(
            name="kernel2_2", shape=(3,3,64,128),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv2_2 = tf.nn.conv2d(
            conv2_1, kernel2_2,
            strides=conv_strides, padding=padding, name="conv2_2"
        )
        conv2_2 = tf.nn.relu(conv2_2)

        pool2 = tf.nn.max_pool(
            conv2_2, ksize=pooling_strides, strides=pooling_strides,
            padding=padding, name="pool2"
        )

        # block 3
        # input: 56, 56, 128
        # output: 28, 28, 256
        kernel3_1 = tf.get_variable(
            name="kernel3_1", shape=(3,3,128,256),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv3_1 = tf.nn.conv2d(
            pool2, kernel3_1,
            strides=conv_strides, padding=padding, name="conv3_1"
        )
        conv3_1 = tf.nn.relu(conv3_1)

        kernel3_2 = tf.get_variable(
            name="kernel3_2", shape=(3,3,128,256),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv3_2 = tf.nn.conv2d(
            conv3_1, kernel3_2,
            strides=conv_strides, padding=padding, name="conv3_2"
        )
        conv3_2 = tf.nn.relu(conv3_2)

        pool3 = tf.nn.max_pool(
            conv3_2, ksize=pooling_strides, strides=pooling_strides,
            padding=padding, name="pool3"
        )

        # block 4
        # input: 28, 28, 256
        # output: 14, 14, 512
        kernel4_1 = tf.get_variable(
            name="kernel4_1", shape=(3,3,256,512),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv4_1 = tf.nn.conv2d(
            pool3, kernel4_1,
            strides=conv_strides, padding=padding, name="conv4_1"
        )
        conv4_1 = tf.nn.relu(conv4_1)

        kernel4_2 = tf.get_variable(
            name="kernel4_2", shape=(3,3,256,512),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv4_2 = tf.nn.conv2d(
            conv4_1, kernel4_2,
            strides=conv_strides, padding=padding, name="conv4_2"
        )
        conv4_2 = tf.nn.relu(conv4_2)

        pool4 = tf.nn.max_pool(
            conv4_2, ksize=pooling_strides, strides=pooling_strides,
            padding=padding, name="pool4"
        )

        # block 5
        # input: 14, 14, 512
        # output: 7, 7, 512
        kernel5_1 = tf.get_variable(
            name="kernel5_1", shape=(3,3,512,512),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv5_1 = tf.nn.conv2d(
            pool4, kernel5_1,
            strides=conv_strides, padding=padding, name="conv5_1"
        )
        conv5_1 = tf.nn.relu(conv5_1)

        kernel5_2 = tf.get_variable(
            name="kernel5_2", shape=(3,3,512,512),
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        conv5_2 = tf.nn.conv2d(
            conv5_1, kernel5_2,
            strides=conv_strides, padding=padding, name="conv5_2"
        )
        conv5_2 = tf.nn.relu(conv5_2)

        pool5 = tf.nn.max_pool(
            conv5_2, ksize=pooling_strides, strides=pooling_strides,
            padding=padding, name="pool5"
        )

        # flatten
        # input: 7, 7, 512
        # output: ?, 25088
        flatten = tf.reshape(pool5, [-1])

        # fc1 layer
        # input: ?, 25088
        # otput: ?, 4096
        W1 = tf.get_variable(
            name="W1", shape=(25088, 4096),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b1 = tf.get_variable(
            name="b1", shape=(4096),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        fc1 = tf.matmul(flatten, W1) + b1

        # fc2 layer
        # input: ?, 4096,
        # output: ?, 4096
        W2 = tf.get_variable(
            name="W2", shape=(4096, 4096),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b2 = tf.get_variable(
            name="b2", shape=(4096),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        fc2 = tf.matmul(fc1, W2) + b2

        # prediction layer
        # input: ?, 4096
        # output: ?, num_classes
        W3 = tf.get_variable(
            name="W3", shape=(4096, num_classes),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b3 = tf.get_variable(
            name="b3", shape=(num_classes),
            initializer=tf.contrib.layers.xavier_initializer()
        )

        hypothesis = tf.matmul(fc2, W3) + b3
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        with graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                avg_loss = 0

                for epoch in range(self.epochs):
                    for image_batch, class_label_batch in self.batch_generator():
                        feed_dict={X_raw: image_batch, Y:class_label_batch}
                        l, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
                        avg_loss = l/self.total_batch
                    print("Epoch: {}, loss: {}".format(epoch+1, avg_loss))
                
                saver = tf.train.Saver()
                saver.save(sess, "model/model.ckpt")

def gen(numList):
    for i in range(0, len(numList), 10):
        batch = numList[i:i+10]
        yield batch

# a = [i for i in range(100)]

# for b in gen(a):
#     print(b)

a = np.array(pilimg.open("L2.png"))
print(a.shape)