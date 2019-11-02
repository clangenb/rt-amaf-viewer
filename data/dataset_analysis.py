#########################################################################
# Intended for usage in Python Console to get some ideas about the data
#########################################################################
import csv
import numpy as np
import matplotlib.pyplot as plt

import data.mirex_data_handlers as mh

comp_path = '../features/ComParE_2016_fsize500_fstep100/2.csv'
mir_path = '../mirexdatabase/default_features/2.csv'
arousal_std = '../mirexdatabase/annotations/arousal_cont_std.csv'
valence_std = '../mirexdatabase/annotations/valence_cont_std.csv'


def get_header_and_features():
    header_comp = next(csv.reader(open(comp_path), delimiter=';'))
    header_mir = next(csv.reader(open(mir_path), delimiter=';'))

    features_comp = np.genfromtxt(comp_path, delimiter=';', skip_header=1)
    features_mir = np.genfromtxt(mir_path, delimiter=';', skip_header=1)

    return header_comp, header_mir, features_comp, features_mir


def get_matching_strings(string_list, string):
    return [s for s in string_list if string in s]


def plot_statistics():
    # y_a = mh.get_labels_arousal().flatten() / 1.4995299633862431
    y_a = mh.get_std_arousal().flatten() / 1.4995299633862431
    # y_v = mh.get_labels_valence().flatten() / 1.358889011041143
    y_v = mh.get_std_valence().flatten() / 1.358889011041143

    # y_a = mh.get_labels_arousal().mean(axis=1)
    # y_v = mh.get_labels_valence().mean(axis=1)

    h, xbins, ybins = np.histogram2d(y_v, y_a, normed=False, bins=20)
    fig, ax = plt.subplots(1, 1)
    ax.set_title('IOD Distribution')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    x = np.linspace(xbins[0], xbins[-1], len(h))
    y = np.linspace(ybins[0], ybins[-1], len(h))
    plt.contourf(x, y, h, cmap='jet')
    ax.set_xlim([xbins[0], 0.4])
    ax.set_ylim([ybins[0], 0.4])
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.5, ybins[-1]])
    cb = plt.colorbar()
    #
    # x = np.linspace(-0.3, 0.4)
    # y = x + 0.02
    #
    # lines = [Line2D([0], [0], color='brown')]
    # ax.plot(x, y, color='brown')
    # ax.legend(lines, ['Predictions'])

    cb.ax.set_title('Samples/Bin', fontsize=10)
    plt.savefig('../iod_distro.pdf', format='pdf')


def plot_multitask_predictions():
    x = np.linspace(-0.3, 0.4)
    y = x
    plt.plot(x, y)


def compute_irreducible_error(file_id):
    """
     Computes the irreducible error, which is the V(Y|x) == V(Y) if x is measured
    :param file_id: file containing the standard deviation of the assigned labels
    :return: Irreducible error in terms of RMSE
    """
    if file_id == 0:
        file_std = arousal_std
    else:
        file_std = valence_std

    std = np.genfromtxt(file_std, delimiter=',', skip_header=1)
    std = std[:, 2]*1000    # first column song ID, first entry at 15ms skipped
    ire = np.sqrt(np.mean(np.square(std)))

    return ire
