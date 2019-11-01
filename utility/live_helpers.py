#################################################
# Helper functions used in the live environment
#################################################
import numpy as np


def make_feature_vector_from_bytecode_string(smileout):
    """
    :param smileout: bytestream containing numbers for features separated by spaces
    :return: np.array containing floats of shape (1,1,num_features)
    """
    cout = smileout.decode()
    feature_list = cout.split()

    try:
        return np.asarray([float(i) for i in feature_list], dtype=float)
    except ValueError:
        # if its not floats its the feature_list -> return it as strings
        return feature_list


def make_feature_list_from_smileout(smileout):
    cout = smileout.decode()
    feature_names = cout.split()
    print('Feature List Length: ', len(feature_names))
    return feature_names


def get_feature_index_from_string(feature_name, feature_list):
    match = [s for s in feature_list if feature_name in s]
    print(match, feature_name)

    assert (len(match) != 0), "No Matches Found"
    assert (len(match) == 1), 'More than one entry for {}, be more specific!'.format(feature_name)

    return feature_list.indexOf(match)


def normalize_features(features, means, var):
    """ normalize features according to the distribution obtained in training """
    f = features.reshape(-1, features.shape[-1])
    f = (f - means) / var
    return f.reshape((1, 1, features.shape[-1]))


def adjust_mean_var(x_n, mean_old, var_old, decay):
    """ Gradually adapt the means, vars in order adapt to live test scenario"""
    mean_new = decay*mean_old + (1 - decay)*x_n
    var_new = np.sqrt(decay*np.square(var_old) + (1 - decay)*np.square(x_n - mean_new))
    return mean_new, var_new
