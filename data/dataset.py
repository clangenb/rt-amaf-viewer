from __future__ import generators
import numpy as np
import math

import data.datahelpers as dh

# Static parameters
PATH_TO_CSV = "../data/"      # default directory of the feature files for training
VALIDATION_SET = 1
DEVELOPMENT_SET = 0


class Dataset:

    def __init__(self, feature_sets, filenames):
        self.train = None
        self.test = None
        self.feature_sets = feature_sets.copy()
        self.file_names = filenames
        self.size = self.feature_sets.shape[0]

    def split(self, batch_size, sequence_length, test_size=0.2, shuffle=True, include_std=False):
        z = list(zip(self.file_names, self.feature_sets))

        if shuffle:
            np.random.shuffle(z)

        self.file_names, self.feature_sets = zip(*z)

        self.feature_sets = np.array(self.feature_sets)
        print('Feature Sets Shape: ', self.feature_sets.shape)

        train_files = self.file_names[int(self.size*test_size):]
        test_files = self.file_names[:int(self.size*test_size)]

        train_sets = self.feature_sets[int(self.size*test_size):, :, :]
        test_sets = self.feature_sets[:int(self.size*test_size), :, :]

        self.train = Iterator(train_sets, batch_size, sequence_length, include_std=include_std)
        self.test = Iterator(test_sets, batch_size, sequence_length, include_std=include_std)

        print('Train Files: ', train_files)
        print('Test Files ', test_files)

        return train_files, test_files

    def get_train_labels_mean(self):
        """ Return the label means for the average predictor """
        _, labels = self.train.get_all_batches()
        mean_arousal = np.mean(labels[:, :, 0])
        mean_valence = np.mean(labels[:, :, 1])
        return mean_arousal, mean_valence


class MirexDataSplit:
    def __init__(self, train_sets, test_sets, batch_size, sequence_length, include_std=False):
        self.train = Iterator(train_sets.copy(), batch_size, sequence_length, include_std=include_std)
        self.test = Iterator(test_sets.copy(), batch_size, sequence_length, include_std=include_std)

    def get_train_labels_mean(self):
        """ Return the label means for the average predictor """
        _, labels = self.train.get_all_batches()
        mean_arousal = np.mean(labels[:, :, 0])
        mean_valence = np.mean(labels[:, :, 1])
        return mean_arousal, mean_valence


class Iterator:

    def __init__(self, feature_sets, batch_size, sequence_length, include_std=False):
        # self.batches.shape == [num_songs, frames/song, num_features + labels + std]
        print('Iterator: ', feature_sets.shape)
        self.batches = feature_sets
        self.batch_size = batch_size
        self.num_songs = self.batches.shape[0]
        self.num_frames = self.batches.shape[1]
        self.num_features = self.batches.shape[2] - 4
        # we need to floor as dense layers do not work with variable size thus a possible last partly filled batch will
        # be discarded
        self.num_batches = math.floor(self.num_songs / self.batch_size)
        print('Num batches', self.num_batches)
        self.sequence_length = sequence_length
        self.num_sequences = int(math.ceil(self.num_frames / sequence_length))
        self.sequences = None
        self.include_std = include_std

        self.batch_index = 0
        self.seq_index = 0
        # features of current sequence
        self.features = []
        self.labels = []

    # initialize iterator for next batch
    def next_batch(self):
        if self.batch_index + self.batch_size < self.num_songs + 1:
            next_index = self.batch_index + self.batch_size
        else:
            next_index = self.num_batches * self.batch_size

        batch = self.batches[self.batch_index:next_index, :, :]

        self.features, self.labels = dh.create_features_and_labels(batch, self.include_std)
        self.batch_index = next_index % (self.num_batches * self.batch_size)
        self.seq_index = 0
        self.sequences = self.sequence_iterator()

    def get_all_batches(self):
        return dh.create_features_and_labels(self.batches, self.include_std)

    # Return the time_frames that are currently fed into the RNN
    def sequence_iterator(self):
        for _ in range(self.num_sequences):
            if self.seq_index + self.sequence_length < self.num_frames:
                next_index = (self.seq_index + self.sequence_length)
            else:
                next_index = self.num_frames

            curr_features = self.features[:, self.seq_index:next_index, :]
            curr_labels = self.labels[:, self.seq_index:next_index, :]

            self.seq_index = next_index % self.num_frames
            yield curr_features, curr_labels

    def normalize_mode_train(self):
        flat = self.features.reshape(-1, self.num_features)
        means = flat.mean(axis=0)
        variances = flat.std(axis=0)

        f_normed = dh.normalize(flat, means, variances)

        self.features = f_normed.reshape(-1, self.num_frames, self.num_features)
        return means, variances

    def normalize_mode_test(self, means, variances):
        flat = self.features.reshape(-1, self.num_features)
        f_normed = dh.normalize(flat, means, variances)

        self.features = f_normed.reshape(-1, self.num_frames, self.num_features)

    # At the beginning of each epoch shuffle the training sets.
    # Randomized batches increase convergence behaviour
    def shuffle(self):
        self.batch_index = 0
        np.random.shuffle(self.batches)
