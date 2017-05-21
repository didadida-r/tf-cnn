'''@file lstm.py
The LSTM neural network classifier'''

import seq_convertors
import tensorflow as tf
from classifier import Classifier
from layer import FFLayer
from activation import TfActivation
import inspect

class LSTM(Classifier):
    '''This class is a graph for lstm neural nets.'''

    def __init__(self, output_dim, num_layers, num_units, activation,
                 layerwise_init=True):
        '''
        DNN constructor

        Args:
            output_dim: the DNN output dimension
            num_layers: number of hidden layers
            num_units: number of hidden units
            activation: the activation function
            layerwise_init: if True the layers will be added one by one,
                otherwise all layers will be added to the network in the
                beginning
        '''

        #super constructor
        super(LSTM, self).__init__(output_dim)

        #save all the DNN properties
        self.num_layers = num_layers
        print(self.num_layers)
        self.num_units = num_units
        print(self.num_units)
        self.activation = activation
        self.layerwise_init = layerwise_init
        self.layerwise_init = None

    def __call__(self, inputs, seq_length, is_training=False, reuse=False,
                 scope=None):
        '''
        Add the LSTM variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a list containing
                a [batch_size, input_dim] tensor for each time step
            seq_length: The sequence lengths of the input utterances, if None
                the maximal sequence length will be taken
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the name scope

        Returns:
            A triple containing:
                - output logits
                - the output logits sequence lengths as a vector
                - a saver object
                - a dictionary of control operations:
                    -add: add a layer to the network
                    -init: initialise the final layer
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            weights = {'out':
                tf.get_variable('weights_out', [self.num_units, self.output_dim], initializer=tf.contrib.layers.xavier_initializer())
            }
            
            biases = {'out':
                tf.get_variable('biases_out', [self.output_dim], initializer=tf.constant_initializer(0))
            }

            #convert the sequential data to non sequential data
            nonseq_inputs = seq_convertors.seq2nonseq(inputs, seq_length)

            input_dim = nonseq_inputs.shape[1]
            nonseq_inputs = tf.reshape(nonseq_inputs,[-1,11,40])

            n_steps = 11
            nonseq_inputs = tf.transpose(nonseq_inputs, [1, 0, 2])
            
            keep_prob = 1
            # define the lstm cell
            # use the dropout in training mode
            if is_training and keep_prob < 1:
                lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units, forget_bias=0.0, 
                            input_size=None, activation=tf.nn.relu, layer_norm=False, norm_gain=1.0, 
                            norm_shift=0.0, dropout_keep_prob=keep_prob, dropout_prob_seed=None)
                            
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units, forget_bias=0.0, 
                            input_size=None, activation=tf.nn.relu, layer_norm=False, norm_gain=1.0, 
                            norm_shift=0.0, dropout_keep_prob=1, dropout_prob_seed=None)     
            
            # stack the lstm to form multi-layers
            cell = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell]*self.num_layers, state_is_tuple=True)
            
            # print(int(nonseq_inputs.shape[0]))
            # self._initial_state = cell.zero_state(int(nonseq_inputs.shape[0]), tf.float32)
            
            # apply the dropout for the inputs to the first hidden layer
            if is_training and keep_prob < 1:
                nonseq_inputs = tf.nn.dropout(nonseq_inputs, keep_prob)
                
            final_nonseq_inputs = tf.unstack(nonseq_inputs, num=n_steps, axis=0)

            # Get lstm cell output initial_state=self._initial_state,
            outputs, states = tf.contrib.rnn.static_rnn(cell, final_nonseq_inputs, dtype=tf.float32)
            outputs = outputs[-1]
                       
            # Linear activation, using rnn inner loop last output
            logits = tf.matmul(outputs, weights['out']) + biases['out']

            # # if self.layerwise_init:

                # # #variable that determines how many layers are initialised
                # # #in the neural net
                # # initialisedlayers = tf.get_variable(
                    # # 'initialisedlayers', [],
                    # # initializer=tf.constant_initializer(0),
                    # # trainable=False,
                    # # dtype=tf.int32)

                # # #operation to increment the number of layers
                # # add_layer_op = initialisedlayers.assign(initialisedlayers+1).op

                # # #compute the logits by selecting the activations at the layer
                # # #that has last been added to the network, this is used for layer
                # # #by layer initialisation
                # # logits = tf.case(
                    # # [(tf.equal(initialisedlayers, tf.constant(l)),
                      # # Callable(activations[l]))
                     # # for l in range(len(activations))],
                    # # default=Callable(activations[-1]),
                    # # exclusive=True, name='layerSelector')

                # # logits.set_shape([None, self.num_units])

            if self.layerwise_init:
                #operation to initialise the final layer
                init_last_layer_op = tf.initialize_variables(
                    tf.get_collection(
                        tf.GraphKeys.VARIABLES,
                        scope=(tf.get_variable_scope().name + '/layer'
                               + str(self.num_layers))))

                control_ops = {'add':add_layer_op, 'init':init_last_layer_op}
            else:
                control_ops = None

            #convert the logits to sequence logits to match expected output
            seq_logits = seq_convertors.nonseq2seq(logits, seq_length,
                                                   len(inputs))
   
            #create a saver
            saver = tf.train.Saver()


        return seq_logits, seq_length, saver, control_ops

class Callable(object):
    '''A class for an object that is callable'''

    def __init__(self, value):
        '''
        Callable constructor

        Args:
            tensor: a tensor
        '''

        self.value = value

    def __call__(self):
        '''
        get the object

        Returns:
            the object
        '''

        return self.value