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
        
        X_raw = tf.placeholder(dtype=tf.float32, shape=(224,224,3), name="X_raw")
        X = tf.reshape(X_raw, (-1, 224, 224, 3), name="X")
        Y = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name="Y")

        conv1 = Convnet(X, input_channels=3, output_channels=64, name="conv1_1")

        with graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for image_batch, class_label_batch in self.batch_generator():
                    feed_dict={X_raw: image_batch, Y:class_label_batch}

def gen(numList):
    for i in range(0, len(numList), 10):
        batch = numList[i:i+10]
        yield batch

# a = [i for i in range(100)]

# for b in gen(a):
#     print(b)

a = np.array(pilimg.open("L2.png"))
print(a.shape)