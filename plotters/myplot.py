from __future__ import division

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib.figure import figaspect
import matplotlib.ticker as tck
from matplotlib.lines import Line2D
import numpy as np

import visualizer.color_base_functions as cbf


SAVE_PATH = '../figures/'
TEMP_PATH = '../figures/tmp/'

epochs = range(5, 100, 5)
ep_labels = ['Epoch' + str(i) for i in epochs]


class PredictPlot:
    def __init__(self, model, song_no):
        self.predictions = []
        path = '../metrics/{}/'.format(model)
        xy = []
        for i in epochs:
            file = 'ep_{}_xy.csv'.format(i)
            xy.append(np.loadtxt(path + file, delimiter=','))

        xy = np.array(xy)
        self.predictions = xy[:, :int(xy.shape[1]/2), :]
        self.labels = xy[0, int(xy.shape[1]/2):, :]
        # print('predictions shape:', self.predictions.shape)
        # print('labels shape: ', self.labels.shape)

        fig, ax = plt.subplots(1, 1)
        plot_range = range(0, 20, 5)
        for i in plot_range:
            ax.plot(self.predictions[i, song_no, :])

        ax.plot(self.labels[song_no, :])
        leg = [ep_labels[i] for i in plot_range]
        leg.append('Labels')
        ax.legend(leg, fontsize='x-small')

        self.setup_aesthetics(ax=ax)

    def setup_aesthetics(self, ax):
        ax.set_xlabel('Frames')
        ax.set_title('Learning Predictions')


class LossPlot:
    def __init__(self, model):
        fig, ax = plt.subplots()
        train_path = '../metrics/{}/train_loss.csv'.format(model)
        test_path = '../metrics/{}/test_loss.csv'.format(model)

        train_l = np.loadtxt(train_path, delimiter=',')
        test_l = np.loadtxt(test_path, delimiter=',')
        train_l /= np.max(train_l)
        test_l /= np.max(test_l)

        # test_min = np.amin(test_l)
        # x_min = np.argmin(test_l)
        # print(x_min)
        x = np.linspace(0, len(train_l), len(test_l))

        print('Shape', train_l.shape)

        ax.grid(True)
        ax.plot(train_l)
        ax.plot(x, test_l)

        # ax.axhline(test_min)
        # ax.axvline(x_min)

        ax.set_title('Losses')
        ax.legend(['Train Loss', "Test Loss"], fontsize='small')

        ax.set_xlabel('Epochs')


class ColorPalette:
    def __init__(self):
        plt.close()
        fix, axes = plt.subplots(2, 2)

        blue = cbf.get_emotion_color(-0.1, -0.1)
        green = cbf.get_emotion_color(-0.3, 0.1)

        yellow = cbf.get_emotion_color(0.4, 0.4)
        violett = cbf.get_emotion_color(0.5, -0.3)

        green_c = cbf.get_complementary_color(yellow)

        print('Blue:', rgb_to_hsv(blue))
        print('Green:', rgb_to_hsv(green))
        print('Yellow:', rgb_to_hsv(yellow))
        print('Violett:', rgb_to_hsv(violett))

        blue_m = np.clip(cbf.gaussian_color_matrix(blue, std=0.1, size=(4, 4)), a_min=0, a_max=1)
        green_m = np.clip(cbf.gaussian_color_matrix(green, std=0.1, size=(4, 4)), a_min=0, a_max=1)
        green_c_m = np.clip(cbf.gaussian_color_matrix(green_c, std=0.1, size=(4, 4)), a_min=0, a_max=1)
        yellow_m = np.clip(cbf.gaussian_color_matrix_rand_hvs_only(yellow, std=0.08, size=(4, 4)), a_min=0, a_max=1)
        violett_m = np.clip(cbf.gaussian_color_matrix(violett, std=0.4, size=(4, 4)), a_min=0, a_max=1)

        colors = [violett_m, yellow_m, green_c_m, blue_m]

        i = 0

        for ax in axes.flatten():
            ax.imshow(colors[i])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1

        plt.savefig('../palettes_dyadic.pdf', format='pdf')


class ErrorDistribution:
    def __init__(self, error_a, error_v, angle, angle_no_small):
        plt.close()
        w, h = figaspect(1/2.9)
        fix, axes = plt.subplots(1, 3, figsize=(w, h))
        self.fontsize_title = 20
        self.fontsize_label = 15
        plt.tight_layout(pad=3.1, w_pad=3)

        self.nbins = 32

        self.n1 = self.plot_error_annotation(axes[0], data=error_a, title='Arousal')
        self.n2 = self.plot_error_annotation(axes[1], data=error_v, title='Valence')
        self.plot_error_angle(axes[2], data=angle, title='Emotion Angle')
        # self.plot_error_angle(axes[3], data=angle_no_small, title='Emotion Angle No Small')

        # axes[0].axvline(x=-0.225, color='red', linestyle='dashed')
        # axes[0].axvline(x=0.225, color='red', linestyle='dashed')
        # axes[1].axvline(x=-0.230, color='red', linestyle='dashed')
        # axes[1].axvline(x=0.230, color='red', linestyle='dashed')

        for ax in axes:
            ax.set_ylim(axes[2].get_ylim())
            ax.tick_params(axis='both', which='major', labelsize=15)

        plt.savefig('../error_stats_angle.pdf', format='pdf')
        plt.show()


    def plot_error_annotation(self, ax, data, title):
        n, _, _ = ax.hist(data, bins=self.nbins)
        ax.set_title(title, fontsize=self.fontsize_title)
        ax.set_ylabel('samples/bin', fontsize=self.fontsize_label)
        ax.set_xlabel('Error', fontsize=self.fontsize_label)
        ax.set_xlim(-.5, 0.5)
        ax.set_xticks([-.5, -.25, 0, 0.25, 0.5])
        ax.set_xticklabels(['', -0.25, 0, 0.25, ''])

        return n

    def plot_error_angle(self, ax, data, title):
        ax.hist(data, bins=self.nbins)
        ax.set_title(title, fontsize=self.fontsize_title)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xlabel('Error', fontsize=self.fontsize_label)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticklabels(['-$\pi$', '-$\pi$/2', 0, '$\pi$/2', '$\pi$'])


class AnimationSnapShot:
    def __init__(self):
        plt.close()
        fix, axes = plt.subplots(3, 4)

        blue_m, blue_b, blue_d = self.get_anim_snap(-0.1, -0.1)
        green_m, green_b, green_d = self.get_anim_snap(-0.3, 0.1)
        yellow_m, yellow_b, yellow_d = self.get_anim_snap(0.4, 0.4)
        violett_m, violett_b, violett_d = self.get_anim_snap(0.4, -0.3)

        colors = [blue_m, green_m, yellow_m, violett_m,
                  blue_b, green_b, yellow_b, violett_b,
                  blue_d, green_d, yellow_d, violett_d]
        i = 0
        for ax in axes.flatten():
            ax.imshow(colors[i])
            i += 1

        for ax in axes.flatten():
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        plt.savefig('../anisnap.pdf', format='pdf')

    def get_anim_snap(self, arousal, valence):
        color = cbf.get_emotion_color(arousal, valence)
        h, s, v = rgb_to_hsv(color)
        bright = hsv_to_rgb((h, s/2, v))
        dark = hsv_to_rgb((h, s, v/2))

        color_c = cbf.get_complementary_color(color)
        color_std = cbf.get_emotion_std(arousal, valence)

        c_m = np.clip(cbf.gaussian_color_matrix(color, std=color_std, size=(7, 20)), a_min=0, a_max=1)
        b_m = np.clip(cbf.gaussian_color_matrix(bright, std=color_std, size=(7, 20)), a_min=0, a_max=1)
        d_m = np.clip(cbf.gaussian_color_matrix(dark, std=color_std, size=(7, 20)), a_min=0, a_max=1)

        o = cbf.gaussian_color_matrix_rand_hvs_only(color_c, std=0, size=(3, 3))

        matrix = np.zeros((15, 20, 3))
        matrix[4:11] = c_m
        matrix[10:13, 8:11] = o

        bright_m = np.zeros((15, 20, 3))
        bright_m[4:11] = b_m
        bright_m[10:13, 8:11] = o

        dark_m = np.zeros((15, 20, 3))
        dark_m[4:11] = d_m
        dark_m[10:13, 8:11] = o

        return matrix, bright_m, dark_m
