########################################
# Functions to set up RNN topologies
########################################

import tensorflow as tf
from tf.zoneout_wrapper import ZoneoutWrapper

z_prob_cells = 0.15
z_prob_states = 0.15


def lstm_cell(state_size, name):
    cell = tf.nn.rnn_cell.LSTMCell(state_size, use_peepholes=True, state_is_tuple=True, name=name)
    # It was observed that the training-mode zoneout is better in validation than non-training mode
    # cell = ZoneoutWrapper(cell, zoneout_prob=(z_prob_cells, z_prob_states), is_training=True)

    return cell


def stacked_rnn_cell(rnn_config):
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell(rnn_config.state_size, name='layer' + str(i))
         for i in range(rnn_config.num_layers)], state_is_tuple=True)

    return cell


def cell_array(rnn_config):
    cells = [lstm_cell(rnn_config.state_size, name='layer' + str(i))
             for i in range(rnn_config.num_layers)]

    return cells


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def lstm_network(input_data, init_state, config):
    """ Create an RNN cell composed sequentially of a number of RNNCells """

    multi_rnn_cell = stacked_rnn_cell(config)

    # 'outputs' is a tensor of shape [batch_size, max_time, state_size]
    # 'state' is an N-tuple where N is the number of LSTMCells containing
    lstm_outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                            inputs=input_data,
                                            initial_state=init_state,
                                            dtype=tf.float32)

    d1 = tf.layers.dense(inputs=lstm_outputs, units=config.num_outputs, name='dense_layer')
    outputs = tf.layers.dense(inputs=d1, units=1, name='output_layer')

    state = tf.identity(state, name='current_state')

    return outputs, state


class RnnConfig:
    def __init__(self, state_size, num_layers, num_outputs):
        self.state_size = state_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
