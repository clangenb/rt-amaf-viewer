from __future__ import division
from utility import  datahelpers as dh
import numpy as np

framesize = 300
framestep = 100
path_to_features = "../../data/ComParE_2016_reduced_fsize{}_fstep{}/".format(framesize, framestep)
DEVELOPMENT_SET = 0
num_sets = 20

feature_sets, files = dh.create_featuresets(path_to_features, num_sets, DEVELOPMENT_SET)

z = list(zip(files, feature_sets))
np.random.shuffle(z)

unzipped_names, unzipped_sets, = zip(*z)

shuffle_set = feature_sets.copy()
np.random.shuffle(shuffle_set)

zip_occurs = []
shuffle_occurs = []
orig_occurs = []

print('Zipped Occurs')
for i in range(feature_sets.shape[0]):
    zip_occurs.extend(np.argwhere(np.all(unzipped_sets == feature_sets[i,:,:], axis=(1,2))))
    print(np.argwhere(np.all(unzipped_sets == feature_sets[i,:,:], axis=(1,2))))

print('Shuffle Occurs')
for i in range(feature_sets.shape[0]):
    shuffle_occurs.extend(np.argwhere(np.all(shuffle_set == feature_sets[i,:,:], axis=(1,2))))
    print(np.argwhere(np.all(shuffle_set == feature_sets[i,:,:], axis=(1,2))))

print('Orig Occurs')
for i in range(feature_sets.shape[0]):
    orig_occurs.extend(np.argwhere(np.all(feature_sets == feature_sets[i,:,:], axis=(1,2))))
    print(np.argwhere(np.all(feature_sets == feature_sets[i,:,:], axis=(1,2))))

print('Zipped in sets: ', np.reshape(zip_occurs, -1))
print('Shuffled in sets: ', np.reshape(shuffle_occurs, -1))
print('Orig in sets: ', np.reshape(orig_occurs, -1))

zip_sort = np.sort(zip_occurs, axis=None)
shuffle_sort = np.sort(shuffle_occurs, axis=None)
orig_sort = np.sort(orig_occurs, axis=None)

print('Zipped Sort: ', zip_sort)
print('Shuffle Sort: ', shuffle_sort)
print('Orig Sort: ', orig_sort)

print(np.array_equal(zip_sort, orig_sort))
print(np.array_equal(shuffle_sort, orig_sort))