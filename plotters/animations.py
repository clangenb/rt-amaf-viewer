######################################################
# Different animation classes to plot the predictions
# against time for qualitative performance analysis
#######################################################
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as an
from matplotlib.lines import Line2D

import abc

import visualizer.color.color_base_functions as cbf
from plotters.list_containers import PredictionListContainer


class BaseAnimation(an.TimedAnimation):
    def __init__(self, fig, axes, config_str):
        self.prediction_line = Line2D([], [], color='blue')
        self.song_name = axes[0].text(-1.020, 1.050, '', fontsize=9)

        self.config_str = config_str
        an.TimedAnimation.__init__(self, fig, interval=70, blit=True)

    @abc.abstractmethod
    def _init_draw(self):
        self.song_name.set_text('')
        self.prediction_line.set_data([], [])

    @abc.abstractmethod
    def _draw_frame(self, framedata):
        """ draw the frames """

    @abc.abstractmethod
    def new_frame_seq(self):
        """ Create the iterator for the frame sequence """

    def _setup_aesthetics(self, axes):
        axes[0].add_line(self.prediction_line)
        axes[0].set_xlim(-1, 1)
        axes[0].set_ylim(-1, 1)
        axes[0].spines['left'].set_position('center')
        axes[0].spines['bottom'].set_position('center')
        axes[0].spines['right'].set_color('none')
        axes[0].spines['top'].set_color('none')
        axes[0].text(1.020, 0, 'Valence', horizontalalignment='left')
        axes[0].text(0, 1.050, 'Arousal', rotation='horizontal', horizontalalignment='center')
        axes[0].set_xticklabels([-1, '', -0.5, '', '', '', 0.5, '', 1])
        axes[0].set_yticklabels([-1, '', -0.5, '', '', '', 0.5, '', 1])
        axes[0].get_xaxis().get_major_ticks()[4].set_visible(False)
        axes[0].get_yaxis().get_major_ticks()[4].set_visible(False)


class AnimatedPredictionAndLabelPlotter(BaseAnimation):
    def __init__(self, config_str, files):
        fig, ax = plt.subplots(1, 1)
        super().__init__(fig, [ax], config_str)

        self.label_line = Line2D([], [], color='red')
        self.lists = PredictionListContainer()
        self.files = files
        self._setup_aesthetics([ax])

    def _init_draw(self):
        super()._init_draw()
        self.label_line.set_data([], [])

    def _draw_frame(self, framedata):
        ya, yv, pa, pv = next(self.lists.frame_generator(5, 1))

        self.song_name.set_text('Track No: {}'.format(self.files[0]))
        if len(self.lists.ya) % 60 == 0:
            del self.files[0]

        self.prediction_line.set_data(pv, pa)
        self.label_line.set_data(yv, ya)
        self._drawn_artists = [self.prediction_line, self.label_line]

    def new_frame_seq(self):
        return iter(range(len(self.lists.ya)))

    def _setup_aesthetics(self, axes):
        super()._setup_aesthetics(axes)
        axes[0].add_line(self.label_line)
        axes[0].legend([self.prediction_line, self.label_line], ['Prediction', 'Label'])


class AnimatedPredictionAndColorPlotter(BaseAnimation):
    def __init__(self, config_str, files, interpolation_factor=1, num_plotted_frames=5):
        fig, axes = plt.subplots(1, 3, figsize=(plt.figaspect(1 / 3)))

        self.label_line = Line2D([], [], color='red')
        self.lists = PredictionListContainer()
        self.files = files

        self.im_size = (1, 1, 3)
        self.image_prediction = axes[1].imshow(np.zeros(self.im_size))
        self.image_label = axes[2].imshow(np.zeros(self.im_size))
        self.interpolation = interpolation_factor
        self.num_plotted_frames = num_plotted_frames

        super().__init__(fig, axes, config_str)
        self._setup_aesthetics(axes)

    def _init_draw(self):
        self.prediction_line.set_data([], [])
        self.label_line.set_data([], [])
        # self.image_prediction.set_data(np.zeros(self.im_size))
        # self.image_label.set_data(np.zeros(self.im_size))
        self.song_name.set_text('')

    def _draw_frame(self, framedata):
        ya, yv, pa, pv = next(self.lists.frame_generator(self.num_plotted_frames, 1))

        # self.song_name.set_text('Track No: {}'.format(self.files[0]))
        self.song_name.set_text('{}'.format(self.files[0]))
        if len(self.lists.ya) % (60 * self.interpolation) == 0:
            del self.files[0]

        self.prediction_line.set_data(pv, pa)
        self.label_line.set_data(yv, ya)

        pa = np.mean(pa)
        pv = np.mean(pv)
        ya = np.mean(ya)
        yv = np.mean(yv)

        self.image_prediction.set_data(np.reshape(cbf.get_emotion_color(pa, pv), (1, 1, 3)))
        self.image_label.set_data(np.reshape(cbf.get_emotion_color(ya, yv), (1, 1, 3)))

        self._drawn_artists = [self.prediction_line, self.label_line,
                               self.image_prediction, self.image_label]

    def new_frame_seq(self):
        return iter(range(len(self.lists.ya)))

    def _setup_aesthetics(self, axes):
        super()._setup_aesthetics(axes)
        axes[0].add_line(self.label_line)
        axes[0].legend([self.prediction_line, self.label_line], ['Prediction', 'Label'])

        axes[1].set_title('Prediction Associated Color')
        axes[2].set_title('Label Associated Color')
        for ax in axes[1:]:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        plt.tight_layout(pad=4)
