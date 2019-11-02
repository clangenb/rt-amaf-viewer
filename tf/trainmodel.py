###################################################
# Simple LSTM RNN model that initiates its states
###################################################

import tensorflow as tf

import tf.init_ops
from tf.rnn_networks import lstm_network


class TrainModel:
    def __init__(self, x, state_shape, rnn_config):
        self.init_state, rnn_tuple_state = \
            tf.init_ops.init_and_lstm_tuple_state(state_shape)

        with tf.variable_scope('lstm_network'):
            lstm_out, self.state = lstm_network(x, rnn_tuple_state, rnn_config)

        with tf.name_scope('predictions'):
            self.pred = tf.identity(lstm_out, name='pred')
