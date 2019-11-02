from __future__ import division
from joblib import Parallel, delayed

import tensorflow as tf
import numpy as np

from tfmodel.rnn_networks import RnnConfig, gaussian_noise_layer
from tfmodel.trainmodel import TrainModel
import tfmodel.metrics as metrics
import tfmodel.init_ops

from data.dataset import MirexDataSplit
from utility.time_quantizer import TimeQuantizer
import data.datahelpers as dh


def save_results(config_str, result):
    f = open('../statistics/bests', 'a+')
    f.write(config_str + ' ' + result)
    f.close()


def train(state_size, num_layers, batch_size, num_outputs, sequence_length, affect_type, std):
    ######################################################
    # Preliminary
    ######################################################
    m_list = ['A', 'V']  # define if the model is for arousal or valence
    config_str = 'egemaps_{}_size{}_step{}_bs{}_sl{}_nl{}_ss{}_no{}_std{}'.format(m_list[affect_type],
                                                                                  framesize,
                                                                                  framestep,
                                                                                  batch_size,
                                                                                  sequence_length,
                                                                                  num_layers,
                                                                                  state_size,
                                                                                  num_outputs,
                                                                                  std)

    rnn_config = RnnConfig(state_size=state_size,
                           num_layers=num_layers,
                           num_outputs=num_outputs)

    # Remove possible older graphs
    tf.reset_default_graph()

    # data = Dataset(all_feature_sets, filenames=files)
    # _, test_files = data.split(batch_size, sequence_length, test_size)
    data = MirexDataSplit(train_sets, test_sets, batch_size, sequence_length)
    mean_arousal, mean_valence = data.get_train_labels_mean()

    # Create placeholders: use None instead of batch_size to enable a variable batch size for testing
    x = tf.placeholder(tf.float32, [None, None, data.train.num_features], name='x')
    y = tf.placeholder(tf.float32, [None, None, 2], name='y')

    mode = tf.placeholder(tf.string, name='mode')
    tf_mean_list, tf_std_list = tfmodel.init_ops.mean_std_placeholders()

    ########################################################
    # Define Tensorflow operations aka graph nodes
    ########################################################

    with tf.name_scope('gaussian_input_layer'):
        x = tf.cond(tf.equal(mode, 'train'), lambda: gaussian_noise_layer(x, std), lambda: x)

    with tf.variable_scope('model_{}'.format(m_list[affect_type])):
        model = TrainModel(x, [num_layers, 2, None, state_size], rnn_config)

    with tf.name_scope('error'):
        rmse = metrics.l1diff_rms_error(model.pred, y[:, :, affect_type])

    with tf.name_scope('adam_optimizer'):
        optimizer = tfmodel.init_ops.adam_optimizer(rmse, data.train.num_batches * data.train.num_sequences)

    with tf.name_scope('feature_means_std'):
        tf_means, tf_std = tfmodel.init_ops.mean_std_variable(data.train.num_features)

        mean_op = tf_means.assign(tf.reduce_mean(tf.stack(tf_mean_list), axis=0, name='mean_op'))
        std_op = tf_std.assign(tf.reduce_mean(tf.stack(tf_std_list), axis=0, name='std_op'))

    ############################################################################################################
    # Training & Evaluation
    ############################################################################################################

    # Enable flexible GPU memory allocation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    timer = TimeQuantizer()

    with tf.Session(config=config) as sess:
        # writer enables TensorBoard visualization of the graph
        # Command: tensorboard --logdir=./graphs
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())

        # plotter = MyPlotter()
        best_result = 180
        feature_means = []
        feature_vars = []
        metrix = metrics.MetricsContainer(filenames=test_files)
        metrix.save_filenames('../metrics/{}/'.format(config_str))

        c_state = np.zeros((num_layers, 2, batch_size, state_size))

        for epoch in range(num_epochs):
            epoch_loss = 0
            data.train.shuffle()

            for i in range(0, data.train.num_batches):
                data.train.next_batch()

                mean, var = data.train.normalize_mode_train()
                feature_means.append(mean)
                feature_vars.append(var)

                for seq_x, seq_y in data.train.sequences:
                    # print('Seq_x shape:', seq_x.shape, 'Seq_y shape:', seq_y.shape)
                    _, c_state, rms = \
                        sess.run([optimizer, model.state, rmse],
                                 feed_dict={x: seq_x,
                                            y: seq_y,
                                            model.init_state: c_state,
                                            mode: 'train'})

                    epoch_loss += rms
            metrix.train_loss.append(epoch_loss)

            ############################################################################################################
            # Evaluate the model_type regularly
            ############################################################################################################

            if (epoch + 1) % validation_epochs == 0:
                print('Training time after {} epochs: {} minutes\n'.format(
                    epoch + 1, round(timer.measure_total() / 60, 3)))

                random_loss = 0
                test_loss = 0

                test_pred_a = []
                test_pred_v = []
                test_lab_a = []
                test_lab_v = []

                sess.run([mean_op, std_op], feed_dict={tf_mean_list: feature_means, tf_std_list: feature_vars})
                # print('Feature means:', tf_means.eval())

                c_state = np.zeros((num_layers, 2, batch_size, state_size))

                for i in range(0, data.test.num_batches):
                    data.test.next_batch()
                    data.test.normalize_mode_test(tf_means.eval(), tf_std.eval())

                    for seq_x, seq_y in data.test.sequences:

                        [c_state, rms, p] = \
                            sess.run([model.state, rmse, model.pred],
                                     feed_dict={x: seq_x,
                                                y: seq_y,
                                                model.init_state: c_state,
                                                mode: 'test'})

                        random_loss += np.sqrt(np.mean(np.square(mean_arousal - seq_y[:, :, affect_type])))
                        test_loss += rms
                        metrix.extend_predictions_labels(p, seq_y[:, :, affect_type])

                metrix.test_loss.append(test_loss)
                tot_sequences = data.test.num_sequences * data.test.num_batches
                test_rmse = test_loss / tot_sequences
                random_rmse = random_loss / tot_sequences

                metrix.save_predictions_labels('../metrics/{}/'.format(config_str), epoch+1)
                metrix.flush_predictions_labels()

                print('{} Epoch {}: Test RMSE: {}, Random RMSE: {}'.format(
                    config_str, epoch + 1, test_rmse, random_rmse))

                if test_rmse < best_result:
                    best_result = test_rmse
                    result_str = 'Epoch {}: RMS: {}\n'.format(epoch + 1, test_rmse)
                    save_results(config_str, result_str)

                    saver = tf.train.Saver()
                    saver.save(sess, 'model/' + config_str + '/model.ckpt', global_step=epoch + 1)

                metrix.save_train_test_loss('../metrics/{}/'.format(config_str))
                writer.close()


if __name__ == "__main__":
    framesize = 300
    framestep = 100
    # path_to_features = "../data/compare2016/".format(framesize, framestep)
    path_to_features = "../data/eGeMAPSv01a_fsize300_fstep100"
    # path_to_features = "../data/ComParE_2016_reduced_fsize{}_fstep{}/".format(framesize, framestep)

    DEVELOPMENT_SET = 0
    VALIDATION_SET = 1
    AROUSAL = 0
    VALENCE = 1

    # training specifications
    test_size = 0.2
    num_sets = -1                      # -1 to include all sets
    num_epochs = 100
    batch_size_list = [16]
    sequence_length_list = [10, 30]     # back-propagation: should be bigger than the maximum relevance of past inputs
    validation_epochs = 1
    affect_list = [AROUSAL, VALENCE]
    # all_feature_sets, files = dh.create_featuresets(path_to_features, num_sets, DEVELOPMENT_SET)
    train_sets, train_files = dh.create_featuresets(path_to_features, -1, DEVELOPMENT_SET)
    test_sets, test_files = dh.create_featuresets(path_to_features, -1, VALIDATION_SET)

    # RNN specifications
    state_size_list = [40, 60]
    num_layers_list = [2]
    num_outputs_list = [10]
    std_list = [0.1, 0.2, 0.3]

    # train(state_size=80,
    #       num_layers=2,
    #       batch_size=16,
    #       all_feature_sets=feature_sets,
    #       num_outputs=10,
    #       sequence_length=30,
    #       affect_type=AROUSAL)

    Parallel(n_jobs=1)(delayed(train)(state_size=size,
                                      num_layers=num,
                                      batch_size=bs,
                                      num_outputs=no,
                                      sequence_length=sl,
                                      affect_type=affect,
                                      std=std)
                       for size in state_size_list
                       for num in num_layers_list
                       for bs in batch_size_list
                       for no in num_outputs_list
                       for sl in sequence_length_list
                       for affect in affect_list
                       for std in std_list
                       )
