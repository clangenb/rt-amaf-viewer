#################################################################
# Class that sets up all the preliminaries and framework for a
# trained model, such that from the outside only predict can be
# called without worrying about state updates etc.
#################################################################

import tensorflow as tf
import numpy as np

import tf.tensor_loaders as tl
import utility.live_helpers as lh


class TestModel:
    def __init__(self, model, batch_size, namescope):
        self.graph = tf.Graph()

        # Enable flexible GPU memory allocation
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=tf_config)

        with self.graph.as_default():
            print('Restoring Model')
            saver = tf.train.import_meta_graph(model + '.meta')
            saver.restore(self.sess, model)

            self._x, _ = tl.load_xy(self.graph)
            self._init_state = tl.load_init_states(namescope, self.graph)
            self._state = tl.load_current_states(namescope, self.graph)
            self._pred = tl.load_predictions(namescope, self.graph)
            self._mode = tl.load_mode(self.graph)

            means, std = tl.load_means_std(self.graph)
            self.f_means, self.f_std = self.sess.run([means, std])

        self._c_state = np.zeros((2, 2, batch_size, 80))

    def predict(self, x):
        # print('x.shape: ', x.shape)
        self.f_means, self.f_std = lh.adjust_mean_var(x, self.f_means, self.f_std, decay=0.98)
        x = lh.normalize_features(x, self.f_means, self.f_std)

        p, self._c_state = \
            self.sess.run([self._pred, self._state],
                          feed_dict={self._x: x,
                                     self._init_state: self._c_state,
                                     self._mode: 'test'
                                     })
        return p
