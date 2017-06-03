'''@file dnn.py
The DNN neural network classifier'''

import seq_convertors
import tensorflow as tf
from classifier import Classifier
from layer import FFLayer, CnnLayer, LSTMLayer, LSTMLayer2, LSTMLayer3
from activation import TfActivation

class DNN(Classifier):
    '''This class is a graph for feedforward fully connected neural nets.'''

    def __init__(self, output_dim, num_layers, num_units, conf, activation,
                 max_input_length, layerwise_init=True):
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
        super(DNN, self).__init__(output_dim)
        
        # conf
        self.cnn_conf = dict(conf.items('cnn'))
        self.lstm_conf = dict(conf.items('lstm'))
        self.lstm_conf2 = dict(conf.items('lstm2'))
        self.lstm_conf3 = dict(conf.items('lstm3'))
        self.max_input_length = max_input_length
        
        self.lstm_type = int(conf.get('nnet','lstm_type'))
        self.cnn_type = int(conf.get('nnet','cnn_type'))

        #save all the DNN properties
        self.FL_num_layers = num_layers
        # mean the lstm or cnn layers
        num_other_layers = 1
        # contain the FL_hidden layer and lstm/cnn layers
        self.num_layers = self.FL_num_layers + num_other_layers
            
        self.num_units = num_units
        self.activation = activation
        self.layerwise_init = layerwise_init

    def __call__(self, inputs, seq_length, is_training=False, reuse=False,
                 scope=None):
        '''
        Add the DNN variables and operations to the graph

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
            #input layer
            layer = FFLayer(self.num_units, self.activation)
            #output layer
            outlayer = FFLayer(self.output_dim,
                               TfActivation(None, lambda x: x), 0)

            #convert the sequential data to non sequential data
            ## if you wanna use the pure dnn, please uncommit this line 
            #nonseq_inputs = seq_convertors.seq2nonseq(inputs, seq_length)
            
            activations = [None]*self.num_layers
            
            # Define the first hidden layers 
            # # the conv layer
            #cnn_layer = RestNet()
            #cnn_layer = CnnVd6()
            if self.cnn_type == 1:
                print('------The Cnn Config------')
                #convert the sequential data to non sequential data
                nonseq_inputs = seq_convertors.seq2nonseq(inputs, seq_length)
                
                cnn_layer = CnnLayer(self.cnn_conf)
                activations[0] = cnn_layer(nonseq_inputs, is_training, reuse, 'layer0')
            else:
                print("Not using CNN")
            # # the lstm layer, type 1
            if self.lstm_type == 1:
                print('------The LSTM Config------')
                #convert the sequential data to non sequential data
                # the inputs format is: time List(such as 777), each element is 2-D tensor like: batch_size(such as 64) x fre-dim 
                # the nonseq_inputs format is: batch_size x fre-dim, 2-D tensor, here the batch_size = batch_size x time
                nonseq_inputs = seq_convertors.seq2nonseq(inputs, seq_length)
                print('Type1: The lstm data process is the similar to dnn, use the stacking frame and not output state is reused')
                
                lstm_layer = LSTMLayer(self.lstm_conf)
                activations[0] = lstm_layer(nonseq_inputs, is_training, reuse, 'layer0')
            ## the lstm layer, type 2
            elif self.lstm_type == 2:
                print('------The LSTM Config------')
                print('Type2: The lstm data process is totally sequencial')
                
                # here we directly use the seq data, that's para: inputs
                lstm_layer = LSTMLayer2(self.lstm_conf2)
                # the dynamic lstm's output has the format: time x batch_size x feature_dim
                seq_output = lstm_layer(inputs, seq_length, is_training, reuse, 'layer0')
                
                # to connect the dnn, we should tran the seq output to no-seq
                # so we can use directly with dnn
                activations[0] = seq_convertors.seq2nonseq(seq_output, seq_length)
                
            ## the lstm layer, type 3
            elif self.lstm_type == 3:
                print('------The LSTM Config------')
                print('Type3: The lstm data is processed in sub-seq')
                
                # here we directly use the seq data, that's para: inputs
                lstm_layer = LSTMLayer3(self.lstm_conf3, self.max_input_length)
                # the dynamic lstm's output has the format: time x batch_size x feature_dim
                seq_output = lstm_layer(inputs, seq_length, is_training, reuse, 'layer0')
                # to connect the dnn, we should tran the seq output to no-seq
                # so we can use directly with dnn
                
                # Note:
                # the seq_output here should has the first index corresponding to the seq_length
                # shape like: [seq_length, batch-size, output-dim]
                activations[0] = seq_convertors.seq2nonseq(seq_output, seq_length)
            else:
                print("Not using LSTM")
                
            # define the FL hidden layers
            print('------The DNN Config------')
            print("use %d FL hidden layer" % (self.FL_num_layers))
            for l in range(1, self.num_layers):
                print("the " + str(l) + " layer's input is: " + str(activations[l-1].shape))
                activations[l] = layer(activations[l-1], is_training, reuse,
                                       'layer' + str(l))

            if self.layerwise_init:
                #variable that determines how many layers are initialised
                #in the neural net
                initialisedlayers = tf.get_variable(
                    'initialisedlayers', [],
                    initializer=tf.constant_initializer(0),
                    trainable=False,
                    dtype=tf.int32)

                #operation to increment the number of layers
                add_layer_op = initialisedlayers.assign(initialisedlayers+1).op

                #compute the logits by selecting the activations at the layer
                #that has last been added to the network, this is used for layer
                #by layer initialisation
                logits = tf.case(
                    [(tf.equal(initialisedlayers, tf.constant(l)),
                      Callable(activations[l]))
                     for l in range(len(activations))],
                    default=Callable(activations[-1]),
                    exclusive=True, name='layerSelector')

                logits.set_shape([None, self.num_units])
            else:
                logits = activations[-1]

            
            logits = outlayer(logits, is_training, reuse,
                              'layer' + str(self.num_layers))

            if self.layerwise_init:
                #operation to initialise the final layer
                init_last_layer_op = tf.initialize_variables(
                    tf.get_collection(
                        tf.GraphKeys.VARIABLES,
                        scope=(tf.get_variable_scope().name + '/layer'
                               + str(self.FL_num_layers))))

                control_ops = {'add':add_layer_op, 'init':init_last_layer_op}
            else:
                control_ops = None

            #convert the logits to sequence logits to match expected output
            seq_logits = seq_convertors.nonseq2seq(logits, seq_length,
                                                   len(inputs))

            #create a saver
            saver = tf.train.Saver()

        return seq_logits, seq_length, saver, control_ops