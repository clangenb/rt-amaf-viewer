import numpy as np

import visualizer.color.color_base_functions as cb

black = 0x000000
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))


class Curtain:
    def __init__(self):
        self.matrix_size = (15, 20)
        self.curr_on_lines = 0
        self.center_line = 7
        self.on_pixels = []
        self.feature_max = 0
        self.feature_mean = 0

    def get_off_pixels(self, feature):
        self.feature_mean = \
            _update_mean(self.feature_mean, feature, 0.8)

        next_on_lines = self._get_on_rows(self.feature_mean, self.feature_max)

        self.on_pixels = range((7 - next_on_lines)*self.matrix_size[1],
                               (8 + next_on_lines)*self.matrix_size[1])

        off_pixels = set(range(300)).difference(set(self.on_pixels))

        self.curr_on_lines = next_on_lines
        self.feature_max = max([self.feature_max, feature])
        self.feature_max *= 0.995
        return off_pixels

    def _get_on_rows(self, feature, maxima):
        if maxima != 0:
            on_lines = int(np.floor(self.matrix_size[0] / 2 * feature / maxima))
        else:
            on_lines = 1

        return max(on_lines, 1)


def _update_mean(mean_old, x_n, decay):
    mean_new = decay * mean_old + (1 - decay) * x_n
    return mean_new
