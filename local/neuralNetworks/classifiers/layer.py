'''@file layer.py
Neural network layers '''

import tensorflow as tf
import numpy as np

class FFLayer(object):
    '''This class defines a fully connected feed forward layer'''

    def __init__(self, output_dim, activation, weights_std=None):
        '''
        FFLayer constructor, defines the variables
        Args:
            output_dim: output dimension of the layer
            activation: the activation function
            weights_std: the standart deviation of the weights by default the
                inverse square root of the input dimension is taken
        '''

        #save the parameters
        self.output_dim = output_dim
        self.activation = activation
        self.weights_std = weights_std

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        '''
        Do the forward computation
        Args:
            inputs: the input to the layer
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the variable scope of the layer
        Returns:
            The output of the layer
        '''
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):

                stddev = (self.weights_std if self.weights_std is not None
                          else 1/int(inputs.get_shape()[1])**0.5)

                weights = tf.get_variable(
                    'weights', [inputs.get_shape()[1], self.output_dim],
                    initializer=tf.contrib.layers.xavier_initializer())

                biases = tf.get_variable(
                    'biases', [self.output_dim],
                    initializer=tf.constant_initializer(0))

            #apply weights and biases
            with tf.variable_scope('linear', reuse=reuse):
                linear = tf.matmul(inputs, weights) + biases

            #apply activation function
            with tf.variable_scope('activation', reuse=reuse):
                outputs = self.activation(linear, is_training, reuse)

        return outputs

class ConvLayer(object):
    '''This class defines a fully connected feed forward layer'''

    def __init__(self, output_dim, activation):
        '''
        FFLayer constructor, defines the variables
        Args:
            output_dim: output dimension of the layer
            activation: the activation function
            weights_std: the standart deviation of the weights by default the
                inverse square root of the input dimension is taken
        '''

        #save the parameters
        self.output_dim = output_dim
        self.activation = activation
        self.strides = 1

    def __call__(self, inputs, kernel_shape, is_first_layer=False, is_pool=True, is_training=False, reuse=False, scope=None):
        '''
        Do the forward computation
        Args:
            inputs: the input to the conv layer, format shape like [num, x, y, channel]
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the variable scope of the layer
        Returns:
            The output of the layer
        '''
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):

                weights = tf.get_variable(
                    'weights', shape=kernel_shape, dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer_conv2d())

                biases = tf.get_variable(
                    'biases', [kernel_shape[3]],
                    initializer=tf.constant_initializer(0))

            #apply weights and biases
            with tf.variable_scope('linear', reuse=reuse):
                linear = tf.nn.conv2d(inputs, weights, strides=[1, self.strides, self.strides, 1], padding='SAME')
                linear = tf.nn.bias_add(linear, biases)

            #apply activation function
            with tf.variable_scope('activation', reuse=reuse):
                if is_first_layer:
                    linear = tf.reduce_sum(linear, 2, keep_dims=True)     # 将卷积结果按照tensor的第二个维度求和
                outputs = self.activation(linear, is_training, reuse)

            # pool steps
            if is_pool:
                outputs = tf.nn.max_pool(outputs, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],
                                padding='SAME')

        return outputs

class CnnLayer(object):

    def __init__(self):
        pass

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):        
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse): 
            shape = [tf.shape(inputs)[0] , 40, 3, 11]
            inputs_img = tf.reshape(inputs, tf.stack(shape)  ) 
            inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )
            print(shape)

            # 不使用BN层
            # 这里这使用了静态的MFCC特征
            is_BN=False
            conv1 = self.convolution(inputs_img[:,:,:,0:1], 'conv_l1', [9, 9, 1, 128], [1, 1, 1, 1], reuse, is_training, is_BN)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='VALID')
            conv2 = self.convolution(pool1, 'conv_2', [4, 3, 128, 256], [1, 1, 1, 1], reuse, is_training, is_BN)
            shape = conv2.get_shape().as_list()
            outputs = tf.reshape(conv2, tf.stack( [tf.shape(conv2)[0],  shape[1] * shape[2] * shape[3] ] ) )
            
            print("inputs_img.shape" + str(inputs_img.shape))
            print("conv1.shape" + str(conv1.shape))
            print("conv2.shape" + str(conv2.shape))
            print("outputs.shape" + str(outputs.shape))
            
            # conv1 = self.convolution(inputs_img, 'conv_l1', [7, 17, shape[2], 256], reuse, is_training)
            # pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='VALID')
            # conv2 = self.convolution(pool1, 'conv_2', [4, 3, 256, 256], reuse, is_training)
            # shape = conv2.get_shape().as_list()
            # outputs = tf.reshape(conv2, tf.stack( [tf.shape(conv2)[0],  shape[1] * shape[2] * shape[3] ] ) )
            
            # print(inputs_img.shape)
            # print(conv1.shape)
            # print(conv2.shape)
            # print(outputs.shape)
            
            if is_training == False:
                # 从第50帧开始记录
                tf.summary.image('input_img', inputs_img[50:,:,:,:], 10)
                #shape = conv1.shape
                #x1 = tf.reshape(conv1, tf.stack([tf.shape(conv1)[0], shape[1]*16, shape[2]*16, 1]) )
                #tf.summary.image('conv1', x1, 10)
                #tf.summary.image('conv2', conv2, 10)
                #tf.summary.image('conv_img', tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], 1, 1]))
        
        return outputs

    '''
        Forward the Conv process
        Args:
            inputs: the input to the conv layer, format shape like [num, x, y, channel]
            kernel_shape: [f_conv_size, t_conv_size, in_channel, out_channel]
            reuse: wheter or not the variables in the network should be reused
            scope: the variable scope of the layer
        Returns:
            The output of the layer
        '''
    def convolution(self, inputs_img, name, kernel_shape, strides, reuse, is_training, is_BN):
        with tf.variable_scope('parameters_'+name, reuse=reuse):
            n = kernel_shape[0]* kernel_shape[1]* kernel_shape[3]
            weights = tf.get_variable('weights_'+name, kernel_shape,  initializer = tf.contrib.layers.xavier_initializer_conv2d())
            biases = tf.get_variable('biases_'+name,   [kernel_shape[3]],   initializer=tf.constant_initializer(0) )

        with tf.variable_scope('conv_'+name, reuse=reuse):
            conv = tf.nn.conv2d(inputs_img,  weights, strides, padding='VALID')
            if is_BN:
                conv = tf.contrib.layers.batch_norm(conv,
                    is_training=is_training,
                    scope='batch_norm',
                    reuse = reuse)
            hidden = tf.nn.relu(conv + biases)

        return hidden  
