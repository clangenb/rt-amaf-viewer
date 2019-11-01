#########################################################
# Custom TF RNN cell wrapper implementing the zoneout
# regularization technique
#########################################################


import tensorflow as tf

z_prob_cells = 0.15
z_prob_states = 0.15


# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
class ZoneoutWrapper:
    """ Operator adding zoneout to all states (states+cells) of the given cell. """

    def __init__(self, cell, zoneout_prob=(z_prob_cells, z_prob_states), is_training=True, seed=None):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if isinstance(zoneout_prob, float) and not (0.0 <= zoneout_prob <= 1.0):
            raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                             % zoneout_prob)
        self._cell = cell
        self._zoneout_prob = zoneout_prob
        self._seed = seed
        self.is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if not isinstance(self.state_size, tuple):
            raise TypeError("LSTM cell state must be tuple.")
        if len(tuple(self._zoneout_prob)) != 2:
            raise ValueError("zone_out must be a tuple of length 2.")

        output, new_state = self._cell(inputs, state, scope)

        if self.is_training:
            new_state = tuple((1 - state_part_zoneout_prob) * tf.nn.dropout(
                new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                              for new_state_part, state_part, state_part_zoneout_prob in
                              zip(new_state, state, self._zoneout_prob))
        else:
            new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                              for new_state_part, state_part, state_part_zoneout_prob in
                              zip(new_state, state, self._zoneout_prob))

        return output, tf.nn.rnn_cell.LSTMStateTuple(new_state[0], new_state[1])
