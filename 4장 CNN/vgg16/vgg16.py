import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import PIL.Image as pilimg
import PIL

class VGG16():
    def __init__(self, input_image_vector: np.ndarray, epochs: int=20, batch_size: int=100):
        '''
        input_image: PIL Image
        '''
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_image_vector = input_image_vector
        self.total_batch = int(len(input_image_vector)/batch_size)

    def batch_generator():
        for i in range(0, self.total_batch, self.batch_size):
            image_batch = self.input_image_vector[i:i+self.batch_size]
            yield image_batch

    # convolutions layer
    def Convnet(input_tensor, input_channels, output_channels,
                name="Conv", stddev=0.1, padding="SAME", activation="relu",
                kernel_shape=(3,3), pool_shape=(2,2),
                conv_stride=[1,1,1,1], pool_stride=[1,2,2,1]):

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

        # activation
        Conv1 = tf.nn.relu(
            features=Conv1
        )

        # max pooling
        Conv1 = tf.nn.max_pool(
            value=Conv1,
            ksize=(1, *pool_shape, 1),
            strides=pool_stride,
            padding=padding
        )

        return Conv1

    # fully connected layer
    def FCnet():
        pass

    @classmethod
    def train():
        
        X_raw = tf.placeholder(dtype=tf.float32, shape=(224,224,3))
        X = tf.reshape(X_raw, (-1, 224, 224, 3))

def gen(numList):
    for i in range(0, len(numList), 10):
        batch = numList[i:i+10]
        yield batch

# a = [i for i in range(100)]

# for b in gen(a):
#     print(b)

a = np.array(pilimg.open("L2.png"))
print(a.shape)