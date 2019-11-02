################################################
# Helpers to initialize TensorFlow graph nodes
################################################
import tensorflow as tf


def init_and_lstm_tuple_state(shape):
    with tf.name_scope('init_state'):
        i_state = tf.placeholder(tf.float32, shape=shape, name='init_state')
        lstm_state = _lstmtuple_state(i_state, shape[0], namescope='lstm_state_tuple')

    return i_state, lstm_state


def _lstmtuple_state(init_state, num_layers, namescope):
    with tf.name_scope(namescope):
        state_per_layer_list = tf.unstack(init_state, axis=0)
        lstm_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0],
                                           state_per_layer_list[idx][1])
             for idx in range(num_layers)])   # shape[0] == num_layers
    return lstm_tuple_state


def adam_optimizer(cost_function, runs_per_epoch):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step,
                                               decay_steps=runs_per_epoch, decay_rate=0.98)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function, global_step=global_step)

    return optimizer


def mean_std_variable(size):
    means = tf.Variable(tf.zeros([size]), trainable=False, name='means')
    std = tf.Variable(tf.zeros([size]), trainable=False, name='std')
    return means, std


def mean_std_placeholders():
    mean_list = tf.placeholder(tf.float32, [None, None], name='mean_holder')
    std_list = tf.placeholder(tf.float32, [None, None], name='std_holder')
    return mean_list, std_list
