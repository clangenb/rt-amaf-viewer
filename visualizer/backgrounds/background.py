import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import visualizer.color.utils as cbf
from visualizer.backgrounds.curtain import Curtain

black = 0x000000

######################################################################
# Background class to manipulate backgrounder pixel based on openSmile's
# extracted Entropy and Centroid features
######################################################################


class Backgrounder:
    def __init__(self, std, matrix_size):
        self.off_pixels = []
        self.off_ratio = 0
        self.matrix_size = matrix_size

        self.gaussian_mask = cbf.gaussian_color_matrix((0, 0, 0), std=std, size=matrix_size)
        self.curtain = Curtain()

        # store the previous values for smooting
        self._s = 0
        self._v = 0

        # Feature statistics for background modulations
        self.centroid_max = 1
        self.rms_max = 1
        self.centroid_mean = 1
        self.rms_mean = 1
        self.decay = 0.95

    def switch_random_lights_off(self, feature, maxima):
        off_ratio = 1 - feature / maxima
        if abs(off_ratio - self.off_ratio) > 0.5:
            # only update if the difference is big enough to prevent epileptics
            self.off_pixels = np.random.randint(0, 299, int(off_ratio*300))
        return list(self.off_pixels)

    def modulate_color(self, base_color, centroid, rms, entropy):
        self.centroid_max *= 0.99
        self.rms_max *= 0.99

        #print('centroid: ', self.centroid_max)

        h, s, v = rgb_to_hsv(base_color)
        if centroid > 0:
            s *= self.get_s_factor(centroid, self.centroid_mean, self.centroid_max)
        if rms > 0:
            v *= self.get_v_factor(rms, self.rms_mean, self.rms_max)

        h, s, v = clamp_hsv(h, s, v)
        # self._s = _update_mean(self._s, s, 0.99)
        # self._v = _update_mean(self._v, v, 0.9999)
        self._s = _update_mean(self._s, s, 0.75)
        self._v = _update_mean(self._v, v, 0.75)

        if centroid > self.centroid_max:
            self.centroid_max = centroid
        if rms > self.rms_max:
            self.rms_max = rms

        self.centroid_mean = _update_mean(self.centroid_mean, centroid, self.decay)
        self.rms_mean = _update_mean(self.rms_mean, rms, self.decay)
        off_pixels = self.curtain.get_off_pixels(entropy)
        c = np.reshape([h, self._s, self._v], [-1])
        # c = np.reshape([h, (s+ self._s)/2, (v+ self._v)/2], [-1])
        # print('hsv: ' , c)
        return cbf.to_hex_array(np.add(hsv_to_rgb(c), self.gaussian_mask)), off_pixels
        # return cbf.to_hex_array(np.add(c, self.gaussian_mask) * np.mean([(1 - s), v])), off_pixels

    def get_v_factor(self, x_n, x_mean, x_max):
        if x_n > x_mean:
            return 1 + x_n / x_max
        else:
            return x_n / x_max

    def get_s_factor(self, x_n, x_mean, x_max):
        # white is with saturation 0 thus reverse the calculation
        if x_n > x_mean:
            return x_n / x_max
        else:
            return 1 + x_n / x_max

    def update_gaussian_mask(self, arousal, valence):
        # print('STD', self.std)
        std = cbf.get_emotion_std(arousal, valence)
        r = std / np.max(self.gaussian_mask)

        self.gaussian_mask *= r


def _update_mean(mean_old, x_n, decay):
    return decay*mean_old + (1 - decay)*x_n


def clamp_hsv(h, s, v):
    if h > 1:
        h = 1
    if s > 1:
        s = 1
    if v > 0.1:
        v = 0.1
    if s < 0.3:   # it's no fun if everything's just white
        s = 0.3
    return h, s, v
