import numpy as np
from pandas import DataFrame
import os

from utility.mkdir_p import mkdir_p
# Default paths to the mirexdatabase files
filename_metadata = "../mirexdatabase/annotations/songs_info.csv"
file_arousal = "../mirexdatabase/annotations/arousal_cont_average.csv"
file_arousal_std = "../mirexdatabase/annotations/arousal_cont_std.csv"
file_valence = "../mirexdatabase/annotations/valence_cont_average.csv"
file_valence_std = "../mirexdatabase/annotations/valence_cont_std.csv"

path_to_default_features = "../mirexdatabase/default_features/"


def load_csv_directory(csv_directory=path_to_default_features):
    """
    :return: Numpy matrix containing the features Shape: [num_files, num_rows, num_columns]
    """
    feature_sets = []
    # sorted list of files in csv directory
    list_csv = os.listdir(csv_directory)
    list_csv.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for i in list_csv:
        # print('Load_csv_directory: Loading file', csv_directory+i)
        feature_sets.append(np.genfromtxt(csv_directory + i, delimiter=';', skip_header=1))

    feature_sets = np.stack(feature_sets)
    print('Stacked features shape: ', feature_sets.shape)

    return feature_sets


def generate_metadata(infofile=filename_metadata):
    """
    Fetches es the meta data from the mirex database
    :return: Panda Dataframe containing all the data where text entries have been changed to numeric values
    """
    meta_data = DataFrame.from_csv(infofile)
    meta_data.reset_index(level=0, inplace=True)

    mapping = {'Genre': {'\tBlues\t': 0, '\tClassical\t': 1, '\tCountry\t': 2,
                         '\tElectronic\t': 3, '\tFolk\t': 4, '\tJazz\t': 5, '\tPop\t': 6, '\tRock\t': 7},
               'Mediaeval 2013 set': {'development': 0, 'evaluation': 1}
               }
    meta_data = meta_data.replace(mapping)

    return meta_data


def get_validation_data(dataframe):
    """ Gets the mirex validation set feature files """
    df = dataframe
    eval_set = df.loc[df['Mediaeval 2013 set'] == 1]
    return eval_set


def get_labels_valence(file=file_valence):
    """ Gets the valence labels from the mirex database. """
    valence_labels = np.genfromtxt(file, delimiter=',', skip_header=1)
    return valence_labels[:, 1:]


def get_labels_arousal(file=file_arousal):
    """ Gets the arousal labels from the mirex database. """
    arousal_labels = np.genfromtxt(file, delimiter=',', skip_header=1)
    return arousal_labels[:, 1:]


def get_std_arousal(file=file_arousal_std):
    """ Gets the arousal standard deviation from the mirex database. """
    arousal_std = np.genfromtxt(file, delimiter=',', skip_header=1)
    return arousal_std[:, 1:]


def get_std_valence(file=file_valence_std):
    """ Gets the valence standard deviation from the mirex database. """
    valence_std = np.genfromtxt(file, delimiter=',', skip_header=1)
    return valence_std[:, 1:]


def combine_features_labels_to_csv(path=path_to_default_features, arousal=file_arousal, valence=file_valence,
                                   arousal_std=file_arousal_std, valence_std=file_valence_std):
    """ Combines the features and labels into one file. """
    save_path = '../data/ComParE_2016_fsize200_fstep100/'
    mkdir_p(save_path)

    list_csv = os.listdir(path)
    list_csv.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    arousal = get_labels_arousal(arousal)
    arousal_std = get_std_arousal(arousal_std)
    valence = get_labels_valence(valence)
    valence_std = get_std_valence(valence_std)

    corrupted_list = []

    i = 0
    for file in list_csv:
        feature_set = np.genfromtxt(path + file, delimiter=';', skip_header=1)
        # 148 at 14,8 seconds
        combined = combine_file(feature_set[148::5, 2:],
                                arousal[i, :-1].reshape(-1, 1),
                                valence[i, :-1].reshape(-1, 1),
                                arousal_std[i, :-1].reshape(-1, 1),
                                valence_std[i, :-1].reshape(-1, 1))

        if combined is not None:
            print("Saving file: ", file, 'Number: ', i)
            np.savetxt(save_path + file, combined, delimiter=',')
        else:
            print('Dimensionality Error probably due to corrupted wave in file:', file)
            corrupted_list.append(file)

        i = i+1

    print('Corrupted: ', corrupted_list)


def combine_features_repeated_labels_to_csv(arousal=file_arousal, valence=file_valence,
                                            arousal_std=file_arousal_std, valence_std=file_valence_std):
    """
    Combines the features and labels into one file, while repeating the labels n times with n
    being the number of samples per label.
    """

    save_path = '../data/ComParE_2016_fsize200_fstep100_repeat/'
    path = '../features/ComParE_2016_fsize200_fstep100/'
    mkdir_p(save_path)

    list_csv = os.listdir(path)
    list_csv.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    arousal = get_labels_arousal(arousal)
    arousal_std = get_std_arousal(arousal_std)
    valence = get_labels_valence(valence)
    valence_std = get_std_valence(valence_std)

    corrupted_list = []

    i = 0
    for file in list_csv:
        feature_set = np.genfromtxt(path + file, delimiter=';', skip_header=1)
        # 148 at 14,8 seconds
        combined = combine_file(feature_set[148:, 2:],
                                arousal[i, :-1].repeat(5).reshape(-1, 1),
                                valence[i, :-1].repeat(5).reshape(-1, 1),
                                arousal_std[i, :-1].repeat(5).reshape(-1, 1),
                                valence_std[i, :-1].repeat(5).reshape(-1, 1))

        if combined is not None:
            print("Saving file: ", file, 'Number: ', i)
            np.savetxt(save_path + file, combined, delimiter=',')
        else:
            print('Dimensionality Error probably due to corrupted wave in file:', file)
            corrupted_list.append(file)

        i = i + 1
    print('Corrupted: ', corrupted_list)


def combine_file(feature_set, arousal, valence, arousal_std, valence_std):
    print('feature set shape: ', feature_set.shape)
    print('Labels shape:', arousal.shape)

    try:
        combined = np.concatenate((feature_set,
                                   arousal,
                                   valence,
                                   arousal_std,
                                   valence_std), axis=1)
        print("combined shape: ", combined.shape)
        return combined
    except ValueError:
        return None
