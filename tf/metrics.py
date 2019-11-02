import tensorflow as tf
import numpy as np
from utility.mkdir_p import mkdir_p


def l1diff_rms_error(prediction, label):
    with tf.name_scope('l1diff'):
        pred_f = tf.reshape(prediction, [-1])
        l1diff = tf.subtract(pred_f, tf.reshape(label, tf.shape(pred_f), name='y_flat'), name='l1diff')
    with tf.name_scope('rmse'):
        rmse = tf.sqrt(tf.reduce_mean(tf.square(l1diff)), name='rmse')

    return rmse


def rmse_angle(pred_a, pred_v, label_a, label_v):
    angle_p = tf.angle(tf.complex(pred_a, pred_v))
    angle_y = tf.angle(tf.complex(label_a, label_v))

    _, rmse_v = l1diff_rms_error(pred_v, label_v)
    _, rmse_a = l1diff_rms_error(pred_a, label_a)
    _, rmse_an = l1diff_rms_error(angle_p, angle_y)

    return rmse_an + rmse_a, + rmse_v



############################################################
# Saves the training process to csv files for visualization
#############################################################
class MetricsContainer:
    def __init__(self, filenames):
        self.files = filenames
        self.train_loss = []
        self.test_loss = []
        self.predictions = None
        self.labels = None

    def extend_predictions_labels(self, pred, labels):
        p_tmp = pred.reshape(-1, pred.shape[1])
        if self.predictions is None:
            self.predictions = p_tmp
            self.labels = labels
        else:
            self.predictions = np.concatenate((self.predictions, p_tmp), axis=1)
            self.labels = np.concatenate((self.labels, labels), axis=1)

    def flush_predictions_labels(self):
        self.predictions = None
        self.labels = None

    def save_predictions_labels(self, path, epoch):
        """ Saves predictions and labels into same file with first half of lines being predictions and second labels """
        x = self.predictions.reshape(-1, 60)
        y = self.labels.reshape(-1, 60)

        # print('Pred shape:', x.shape)
        # print('Label shape:', y.shape)

        tmp = np.concatenate([x, y], axis=0)
        # print('Concatenated shape: ', tmp.shape)
        mkdir_p(path)
        save_file = '{}ep_{}_xy.csv'.format(path, epoch)
        np.savetxt(save_file, tmp, delimiter=',')

    def save_train_test_loss(self, path):
        mkdir_p(path)
        save_file = '{}/train_loss.csv'.format(path)
        np.savetxt(save_file, self.train_loss, delimiter=',')
        save_file = '{}/test_loss.csv'.format(path)
        np.savetxt(save_file, self.test_loss, delimiter=',')

    def save_filenames(self, path):
        mkdir_p(path)
        save_file = '{}/test_files.csv'.format(path)
        np.savetxt(save_file, self.files, delimiter=',')
