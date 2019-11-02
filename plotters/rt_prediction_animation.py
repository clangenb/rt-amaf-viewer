import tensorflow as tf
import numpy as np

from data.dataset import Dataset
import data.datahelpers as dh
from tf.testmodel import TestModel
from visualizer.color_base_functions import renormalize_angle

path_to_csv = '../data/ComParE_2016_reduced_fsize300_fstep100/'
good_songs = [44, 49, 12, 112, 88, 39, 172, 143, 143, 142, 300, 665, 724]
bad_songs = [292, 458, 521, 647, 691]
interesting_songs = [462, 622, 729, 704, 810, 899]
files = [292, 88, 665, 458, 521]
song_names = ['Stompin\' Riff Raffs - \nHorror Show', 'Lorenzo\' Music - \nWe All Fall Down',
              'Laché Swing - \nSweet Georgia Brown', 'Nameless Dancers - \nNight Fly',
              'Black Twig Pickers and Steve Gunn - \nOld Strange']
num_plotted_frames = 15

VALIDATION_SET = 1
config_str_a = 'run2_A_size300_step100_bs16_sl10_nl2_ss80_no10_std_0.9'
config_str_v = 'V_size300_step100_bs16_sl30_nl2_ss80_no10'
train_epoch = 50
interp = 10

model_a = 'model/{}/model.ckpt-{}'.format(config_str_a, 60)
model_v = 'model/{}/model.ckpt-{}'.format(config_str_v, 80)

test_size = 1
num_sets = -1  # -1 includes all sets
batch_size = 1
sequence_length = 1  # in online mode only one sample at a time is input
feature_sets, files = dh.create_featuresets(path_to_csv, num_sets, VALIDATION_SET)
# feature_sets = dh.get_featuresets_by_tracklist(path_to_csv, files)
print(feature_sets.shape)

def predict(_):
    data = Dataset(feature_sets, files)
    data.split(batch_size, sequence_length, test_size=test_size, shuffle=False, include_std=False)

    predictor_a = TestModel(model_a, batch_size, 'model_A')
    predictor_v = TestModel(model_v, batch_size, 'model_V')

    rmse_a = []
    rmse_v = []
    angle_err = []
    angle_no_small = []

    # plotter = AnimatedPredictionAndColorPlotter(config_str_a, song_names,
    #                                             interpolation_factor=interp,
    #                                             num_plotted_frames=num_plotted_frames)

    for _ in range(0, data.test.num_batches):
        data.test.next_batch()
        # data.test.normalize_mode_test(predictor_a.f_means, predictor_a.f_std)

        for seq_x, seq_y in data.test.sequences:
            pa = predictor_a.predict(seq_x)
            pv = predictor_v.predict(seq_x)

            ya = seq_y[0,0,0]
            yv = seq_y[0,0,1]

            rmse_a.append((pa - ya) / 1490)
            rmse_v.append((pv - yv) / 1430)

            p_angle = renormalize_angle(np.angle(pa + 1j*pv), deg=False)
            y_angle = renormalize_angle(np.angle(ya + 1j*yv), deg=False)
            err = p_angle - y_angle

            if -np.pi < err < np.pi:
                angle_err.append(err)
                if (abs(ya) > 90) and (abs(yv) > 90):
                    angle_no_small.append(err)
            else:
                angle_err.append(2*np.pi - abs(err))
                if (abs(ya) > 90) and (abs(yv) > 90):
                    angle_no_small.append(2*np.pi - abs(err))

    return rmse_a, rmse_v, angle_err, angle_no_small


    #
    # print('Arousal: Mean: {} Std: {}, kur: {}'.format(np.mean(rmse_a), np.std(rmse_a), scipy.stats.kurtosis(rmse_a, fisher=False)))
    # print('Valence: Mean: {} Std: {}, kur: {}'.format(np.mean(rmse_v), np.std(rmse_v), scipy.stats.kurtosis(rmse_v, fisher=False)))





    # if(interp > 1):
    #     plotter.lists.interpolate(interp=interp)
    # save_str = '../animations/presentation_{}.mp4'.format('final')
    # print('Saving File to:{}'.format(save_str))
    # plotter.save(save_str, fps=2*interp)


if __name__ == '__main__':
    tf.app.run(main=predict)
