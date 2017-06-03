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

class CnnLayer(object):

    def __init__(self, conf):
        self.conf = conf
        self.layers = int(conf['layers'])
        self.freq_dim = int(conf['freq_dim'])
        self.time_dim = int(conf['time_dim'])
        self.input_channel = int(conf['input_channel'])
        self.pool_size = int(self.conf['pool_size'])

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):        
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse): 
            
            # the fitst conv way 
            # Reshape the inputs data, [N, F, 1, T]
            shape = [tf.shape(inputs)[0] , self.freq_dim, self.time_dim, self.input_channel]
            inputs_img = tf.reshape(inputs, tf.stack(shape)  ) 
            inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )     
            print('the inputs_img to conv is : ' + str(shape))
            
            is_BN=False
            for i in range(self.layers):
                kernel = list(eval(self.conf['conv'+ str(i) + '_kernel']))
                strides = list(eval(self.conf['conv'+ str(i) + '_strides']))
                
                print(str(kernel))
                if i == 0:
                    conv = self.convolution(inputs_img, 'ConvL0', kernel, strides, reuse, is_training, is_BN)
                    pool = tf.nn.max_pool(conv, ksize=[1, self.pool_size, 1, 1], strides=[1, self.pool_size, 1, 1], padding='VALID')
                else:
                    conv = self.convolution(conv, 'ConvL'+ str(i), kernel, strides, reuse, is_training, is_BN)
                print('the ' + str(i) + 'th cnn layer')
                print("conv is: " + str(conv.shape))
                print("pool is: " + str(pool.shape))
            shape = conv.get_shape().as_list()
            outputs = tf.reshape(conv, tf.stack( [tf.shape(conv)[0],  shape[1] * shape[2] * shape[3] ] ) )    
            print("the cnn outputs is : " + str(outputs.shape))
            
            if is_training == False:
                # 从第50帧开始记录
                # but we only get the first output channel
                tf.summary.image('input_img', inputs_img[50:,:,:,0:1], 10)
        
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

class LSTMLayer(object):

    def __init__(self, conf):
        self.num_layers = int(conf['num_layers'])
        self.num_units = int(conf['num_units'])
        self.keep_prob = float(conf['keep_prob'])
        self.freq_dim = int(conf['freq_dim'])
        self.time_steps = int(conf['time_steps'])   # here we only use the input frame and the future frame
        #self.context = (self.time_steps-1) * 2 + 1      # the inputs is sliced in feature process
        
        if conf['layer_norm'] == 'True':
            self.layer_norm = True
        else:
            self.layer_norm = False
        if conf['is_tf_lstm'] == 'True':
            self.is_tf_lstm = True
            self.frequency_skip = int(conf['frequency_skip'])
            # the f-step's input size, default 8
            self.feature_size = int(conf['feature_size'])
            # the f-lstm cell number
            self.tf_num_units = int(conf['tf_num_units'])
        else:
            self.is_tf_lstm = False


    def __call__(self, inputs, is_training=False, reuse=False, scope=None):        
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse): 
            # reshape the inputs like the format: [time_steps, batch, fre_dim]
            print("the inputs is: " + str(inputs.shape))
            assert inputs.shape[1] == self.time_steps*self.freq_dim, "the total splice context (context_left + context_right + 1) should equal to " + str(self.time_steps)
            shape = [tf.shape(inputs)[0] , self.time_steps, self.freq_dim]
            inputs_seq = tf.reshape(inputs, tf.stack(shape) ) 
            inputs_seq = tf.transpose(inputs_seq, [1, 0, 2] )
            # do not use the former input frame
            inputs_seq = tf.slice(inputs_seq, [self.time_steps-1, 0, 0], [self.time_steps, -1, -1])
            print("the slice inputs is: " + str(inputs_seq.shape))
            
            # apply the dropout for the inputs to the first hidden layer
            if is_training and self.keep_prob < 1:
                inputs_seq = tf.nn.dropout(inputs_seq, self.keep_prob)
            else:
                self.keep_prob = 1.0

            ## define the tf-lstm layers
            if self.is_tf_lstm:
                tf_lstm_cell = tf.contrib.rnn.TimeFreqLSTMCell(self.tf_num_units, use_peepholes=False,
                                cell_clip=None, initializer=None,
                                num_unit_shards=1, forget_bias=1.0,
                                feature_size=self.feature_size, frequency_skip=self.frequency_skip, feature_dim=self.freq_dim)
    
             # Define the lstm layer
            print('the tensorflow version is:' + tf.__version__)
            if tf.__version__ == '1.1.0':
#                cell = tf.contrib.rnn.MultiRNNCell([ tf.contrib.rnn.LSTMCell(self.num_units, input_size=None,
#                                                       use_peepholes=False, cell_clip=None,
#                                                       initializer=None, num_proj=self.num_proj, proj_clip=None,
#                                                       num_unit_shards=None, num_proj_shards=None,
#                                                       forget_bias=1.0, state_is_tuple=True,
#                                                       activation=tf.nn.relu, reuse=reuse) 
#                                                    for _ in range(self.num_layers)], state_is_tuple=True) 
    
                cell = tf.contrib.rnn.MultiRNNCell([ tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units, forget_bias=1.0, 
                                                        input_size=None, activation=tf.nn.relu, layer_norm=self.layer_norm, norm_gain=1.0, 
                                                        norm_shift=0.0, dropout_keep_prob=self.keep_prob, dropout_prob_seed=None, reuse=reuse)
                                                                for _ in range(self.num_layers)], state_is_tuple=True) 
        
            elif tf.__version__ == '1.0.0':
                ## define the time-process lstm layer
                #    
                # 1. define the basic lstm layer 
                       
                lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units, forget_bias=1.0, 
                        input_size=None, activation=tf.nn.relu, layer_norm=self.layer_norm, norm_gain=1.0, 
                        norm_shift=0.0, dropout_keep_prob=self.keep_prob, dropout_prob_seed=None)
                
                # 2. stack the lstm to form multi-layers
                cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell]*self.num_layers, state_is_tuple=True)
            else:
                print("the tensorflow version's code is not define")
                
            print("the lstm reshape inputs is: " + str(inputs_seq.shape))
            # tran every time_steps to a list element, so the final_noseq_inputs
            # has the format like the List: [steps0, steps1, ..., stepsN]
            # and every steps hase the format: batch x freq-dim  
            final_nonseq_inputs = tf.unstack(inputs_seq, num=self.time_steps, axis=0)
            #print("final_nonseq_inputs:" + str(np.array(final_nonseq_inputs).shape))
            
            # feed the data to lstm layer and output
            if self.is_tf_lstm:
                tf_outputs, tf_states = tf.contrib.rnn.static_rnn(tf_lstm_cell, final_nonseq_inputs, dtype=tf.float32)
                print("the tf-lstm inputs is: " + str(len(tf_outputs)) + " " + str(tf_outputs[0].shape) )
                outputs, states = tf.contrib.rnn.static_rnn(cell, tf_outputs, dtype=tf.float32)
            else:
                outputs, states = tf.contrib.rnn.static_rnn(cell, final_nonseq_inputs, dtype=tf.float32)
                
                
            #outputs, states = tf.contrib.rnn.static_rnn(cell, final_nonseq_inputs, dtype=tf.float32)
            # only get the final steps's output, the format is : batch x lstm_unit_units
            outputs = outputs[-1]
            print("the lstm outputs is: " + str(outputs.shape))
            
            return outputs
        

'''
    For this type of lstm, we directly put the whole utt into the lstm network,
    process the feed-forworad, the code here can be run
    Problem: the decode result is terrible
    
'''
class LSTMLayer2(object):

    def __init__(self, conf):
        self.num_layers = int(conf['num_layers'])
        self.num_units = int(conf['num_units'])
        self.num_proj = int(conf['num_proj'])
        self.keep_prob = float(conf['keep_prob'])
        self.freq_dim = int(conf['freq_dim'])
        
        if conf['layer_norm'] == 'True':
            self.layer_norm = True
        else:
            self.layer_norm = False
        
        ## tf-lstm conf
        if conf['is_tf_lstm'] == 'True':
            self.is_tf_lstm = True
            self.frequency_skip = int(conf['frequency_skip'])
            # the f-step's input size, default 8
            self.feature_size = int(conf['feature_size'])
            # the f-lstm cell number
            self.tf_num_units = int(conf['tf_num_units'])
        else:
            self.is_tf_lstm = False

    def __call__(self, inputs_seq, seq_length, is_training=False, reuse=False, scope=None):        
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse): 
            
            #print(inputs_seq)
            print("the seq inputs is: " + str(len(inputs_seq)))
            print(inputs_seq[1].shape)
            #print(inputs_seq)

            # apply the dropout for the inputs to the first hidden layer
            if is_training and self.keep_prob < 1:
                inputs_seq = tf.nn.dropout(inputs_seq, self.keep_prob)
            else:
                self.keep_prob = 1.0
                
             # Define the lstm layer
            print('the tensorflow version is:' + tf.__version__)
            if tf.__version__ == '1.1.0':
#                cell = tf.contrib.rnn.MultiRNNCell([ tf.contrib.rnn.LSTMCell(self.num_units, input_size=None,
#                                                       use_peepholes=False, cell_clip=None,
#                                                       initializer=None, num_proj=self.num_proj, proj_clip=None,
#                                                       num_unit_shards=None, num_proj_shards=None,
#                                                       forget_bias=1.0, state_is_tuple=True,
#                                                       activation=tf.nn.relu, reuse=reuse) 
#                                                    for _ in range(self.num_layers)], state_is_tuple=True) 
    
                cell = tf.contrib.rnn.MultiRNNCell([ tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units, forget_bias=1.0, 
                                                        input_size=None, activation=tf.nn.relu, layer_norm=self.layer_norm, norm_gain=1.0, 
                                                        norm_shift=0.0, dropout_keep_prob=self.keep_prob, dropout_prob_seed=None, reuse=reuse)
                                                                for _ in range(self.num_layers)], state_is_tuple=True) 
        
            elif tf.__version__ == '1.0.0':
                ## define the time-process lstm layer
                #    
                # 1. define the basic lstm layer 
                       
                lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, input_size=None,
                   use_peepholes=False, cell_clip=None,
                   initializer=None, num_proj=self.num_proj, proj_clip=None,
                   num_unit_shards=None, num_proj_shards=None,
                   forget_bias=1.0, state_is_tuple=True,
                   activation=tf.nn.relu)
                
                # 2. stack the lstm to form multi-layers
                cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell]*self.num_layers, state_is_tuple=True)
            else:
                print("the tensorflow version's code is not define")
            
            ## For the static inputs
            #final_nonseq_inputs = tf.unstack(inputs_seq, num=777, axis=0)
            #outputs, states = tf.contrib.rnn.static_rnn(cell, final_nonseq_inputs, dtype=tf.float32)
            
            ## For the dynamic inputs
            # the inputs_seq: [t1, t2, tN] in List && each t1 is: batch x fre_dim in 2-D Tensor
            # the final_nonseq_inputs: time x batch x fre_dim in 3-D Tensor
            final_nonseq_inputs = tf.stack(inputs_seq, axis=0)
            outputs, states = tf.nn.dynamic_rnn(cell, final_nonseq_inputs, seq_length, time_major=True, dtype=tf.float32)
            
            #print(seq_length)
            #print(len(outputs))
            #print(outputs.shape)
            #outputs = outputs[-1]
            print("the lstm outputs is: " + str(outputs.shape))
            
            return outputs
        
'''
    For this type of lstm, we split every utt to sub-seq with  fix-size, and then sequentially
    feed the sub-seq to the rnn network, only for the sud-seq in the same utt, we keep use the former
    output state
    Problem: 
    1 OOM for the state reuse method 
    
'''
class LSTMLayer3(object):

    def __init__(self, conf, max_input_length):
        self.conf = conf
        self.num_layers = int(conf['num_layers'])
        self.time_steps = int(conf['time_steps'])
        self.num_units = int(conf['num_units'])
        self.num_proj = int(conf['num_proj'])
        self.keep_prob = float(conf['keep_prob'])
        self.freq_dim = int(conf['freq_dim'])
        self.max_input_length = max_input_length
        
        if conf['layer_norm'] == 'True':
            self.layer_norm = True
        else:
            self.layer_norm = False
        
        ## tf-lstm conf
        if conf['is_tf_lstm'] == 'True':
            self.is_tf_lstm = True
            self.frequency_skip = int(conf['frequency_skip'])
            # the f-step's input size, default 8
            self.feature_size = int(conf['feature_size'])
            # the f-lstm cell number
            self.tf_num_units = int(conf['tf_num_units'])
        else:
            self.is_tf_lstm = False

    def __call__(self, inputs_seq, seq_length, is_training=False, reuse=False, scope=None):        
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse): 

            print("the inputs data is: ")
            print("List with len: " + str(len(inputs_seq)) + " each element is 2-D tensor: " + str(inputs_seq[1].shape))

            # apply the dropout for the inputs to the first hidden layer
            if is_training and self.keep_prob < 1:
                inputs_seq = tf.nn.dropout(inputs_seq, self.keep_prob)
            else:
                self.keep_prob = 1.0
                
             # Define the lstm layer
            print('the tensorflow version is:' + tf.__version__)
            if tf.__version__ == '1.1.0':
                cell = tf.contrib.rnn.MultiRNNCell([ tf.contrib.rnn.LSTMCell(self.num_units, input_size=None,
                                                       use_peepholes=False, cell_clip=None,
                                                       initializer=None, num_proj=self.num_proj, proj_clip=None,
                                                       num_unit_shards=None, num_proj_shards=None,
                                                       forget_bias=1.0, state_is_tuple=True,
                                                       activation=tf.nn.relu, reuse=reuse) 
                                                    for _ in range(self.num_layers)], state_is_tuple=True) 
    
#                cell = tf.contrib.rnn.MultiRNNCell([ tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units, forget_bias=1.0, 
#                                                        input_size=None, activation=tf.nn.relu, layer_norm=self.layer_norm, norm_gain=1.0, 
#                                                        norm_shift=0.0, dropout_keep_prob=self.keep_prob, dropout_prob_seed=None, reuse=reuse)
#                                                                for _ in range(self.num_layers)], state_is_tuple=True) 
            elif tf.__version__ == '1.0.0':
                ## define the time-process lstm layer
                #    
                # 1. define the basic lstm layer 
                       
                lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, input_size=None,
                   use_peepholes=False, cell_clip=None,
                   initializer=None, num_proj=self.num_proj, proj_clip=None,
                   num_unit_shards=None, num_proj_shards=None,
                   forget_bias=1.0, state_is_tuple=True,
                   activation=tf.nn.relu)
                
                # 2. stack the lstm to form multi-layers
                cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell]*self.num_layers, state_is_tuple=True)
            else:
                print("the tensorflow version's code is not define")
            
            ## For the static inputs
            # start to forward the data to rnn
            final_inputs = tf.unstack(inputs_seq, num=self.max_input_length, axis=0)
            #print("the final_inputs is: " + str(len(final_inputs)))
            #print(final_inputs[1].shape)

            assert self.max_input_length%self.time_steps==0, "the max_input_length must be divisible by the self.time_steps"
            sub_seq_num = int(self.max_input_length/self.time_steps)
            print("the total sub-seq num is: " + str(sub_seq_num))    
            
            
            
            if self.conf['reuse_sub_seq_state'] == 'True':
                print("reuse the former sub-seq output state")
                sub_state = None
                outputs = []
                for x in range(sub_seq_num):
                    # remerber to resue the variable for different sub-seq in the same utt
                    if x > 0: tf.get_variable_scope().reuse_variables()
                    sub_inputs = final_inputs[x*self.time_steps : (x+1)*self.time_steps]
                    sub_outputs, sub_state = tf.contrib.rnn.static_rnn(cell, sub_inputs, initial_state=sub_state, dtype=tf.float32)
                    outputs += sub_outputs
            else:
                print("not use the former sub-seq output state")
                sub_inputs = []
                # put all the sub-seq into the list and concat in batch
                # this process is like: 
                # tran the data from a list of [20, batch, fre-dim], the list size stands for the sub_seq_num
                # the result has the form like: 20, batch-size*sub_seq_num, fre-dim
                for x in range(sub_seq_num):
                    sub_tmp = final_inputs[x*self.time_steps : (x+1)*self.time_steps]
                    sub_tmp_stack = tf.stack(sub_tmp, axis=0)
                    sub_inputs.append(sub_tmp_stack)
                result = tf.concat(sub_inputs, 1)
                #print(result.shape)
                
                final_inputs = tf.unstack(result, self.time_steps, axis=0)
                    
                print("final_inputs is: " + str(len(final_inputs)))
                outputs, state = tf.contrib.rnn.static_rnn(cell, final_inputs, dtype=tf.float32)
                outputs = tf.stack(outputs, axis=0)
                outputs = tf.reshape(outputs, tf.stack([self.time_steps*sub_seq_num, -1, self.num_proj])) 
                outputs = tf.unstack(outputs, self.time_steps*sub_seq_num, axis=0)

            print("the lstm outputs is: ")
            print("List with len: "+ str(len(outputs)) + " each element is 2-D tensor: " + str(outputs[0].shape))
            
            return outputs
            
            
            
            