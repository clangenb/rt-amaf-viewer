################################################################################
# Methods that are used within the main modules, mainly to read data in order to
# create feature sets
################################################################################

import numpy as np
import os

from data.mirex_data_handlers import  generate_metadata

def create_features_and_labels(feature_sets, include_std=False):
    """
    Splits the feature sets into feature and arousal and valence labels
    :param feature_sets: Shape: [num_songs, num_frames/song, num_features + labels]
    :return: Tuple: (features, (y_arousal, y_valence)
    """

    if not include_std:
        x = feature_sets[:, :, :-4]  # Shape: (num_songs, frames, features)
        y = feature_sets[:, :, -4:-2]  # Shape: (num_songs, frames, 2)
    else:
        x = feature_sets[:, :, :-4]  # Shape: (num_songs, frames, features)
        y = feature_sets[:, :, -4:]  # Shape: (num_songs, frames, 4)

    return x, y * 1000


NUM_GENRES = 8


def create_featuresets(path, num_sets, set_type):
    # get files per genres
    all_files = []
    for i in range(0, NUM_GENRES):
        all_files.extend(get_shuffled_files_per_genre(i, set_type))

    files_to_read = all_files[:num_sets]
    file_paths = [os.path.join(path, str(file) + '.csv') for file in files_to_read]

    return read_all(file_paths), files_to_read

def get_featuresets_by_tracklist(path, tracks):
    file_paths = [os.path.join(path, str(file) + '.csv') for file in tracks]
    return read_all(file_paths)

def get_shuffled_files_per_genre(genre, set_type):
    """
     Gets full (shuffled) file paths of songs matching a respective genre from the mirex data
    :param genre: values from 0 to 8
    :param set_type: development_set = 0, validation_set == 1
    :return: list of full paths to the feature files
    """
    df = generate_metadata()
    if set_type == 2:
        tracks_genre = df.loc[(df['Genre'] == genre)]
    else:
        tracks_genre = df.loc[(df['Genre'] == genre) & (df['Mediaeval 2013 set'] == set_type)]
    ids = tracks_genre.as_matrix(['song_id'])
    np.random.shuffle(ids)

    return ids[:, 0]


def read_all(files_to_read):
    feature_sets = []
    lazy_reader = lazy_file_reader(files_to_read)
    for i in range(len(files_to_read)):
        f_set = next(lazy_reader)
        if f_set is not None:
            print('Shape: ', f_set.shape)
            feature_sets.append(f_set)

    # Shape: [num_songs, num_frames/song, features+labels]
    return np.stack(feature_sets)


def lazy_file_reader(files_to_read):
    files = files_to_read
    file_index = 0
    for f in files:
        feature_set = None
        try:
            feature_set = np.genfromtxt(f, delimiter=',', skip_header=0)
            print('Reading File: ', f.replace('../features/combined/', ''), 'File Index: ', file_index)
        except IOError:
            print('Error: File {} not found'.format(f.replace('../features/combined/', '')))

        file_index = file_index + 1
        yield feature_set


def normalize(f, means, variances):
    features = f.copy()

    for i in range(features.shape[-1]):
        if features[:, i].std() != 0:
            features[:, i] = (features[:, i] - means[i]) / variances[i]
        else:
            print('Feature at index ', i, 'is constant')
    return features


def normalize_tanh(features, means, variances):
    f_norm = normalize(features, means, variances)
    for i in range(features.shape[-1]):
        if features[:, i].std() != 0:
            features[:, i] = 0.5*(np.tanh(0.01*f_norm[:, i]) + 1)

    return features
