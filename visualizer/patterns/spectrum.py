import numpy as np

import visualizer.color_base_functions as cb
import visualizer.pixel_funcs as pxf

black = 0x000000
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))


class CoefficientShower:
    """ Class that displays coefficient magnitudes column wise"""
    def __init__(self, led_strip, coeffs):
        self.matrix_size = (15, 20)
        self.curr_on_lines = 0
        self.strip = led_strip
        if coeffs > 20:
            self.coeff_num = 20
        else:
            self.coeff_num = coeffs
        self.coeffs = np.zeros(coeffs)
        self.coeffs_max = 0
        self.color = blue

        self.on_pixels = []

    def show(self, coeffs):
        self.coeffs_max = max(self.coeffs_max, np.ndarray.max(np.asarray(coeffs)))
        self.on_pixels = []

        for c in range(self.coeff_num):
            on_row = get_on_rows(self.matrix_size, coeffs[c], self.coeffs_max)
            for r in range(on_row):
                px = pxf.to_pixel_no((c, r))
                self.on_pixels.append(px)
                self.strip.setPixelColor(px, self.color)

        self.coeffs_max *= 0.98

    def get_object_pixels(self):
        return self.on_pixels

    def update_color(self, arousal=None, valence=None):
        if(arousal is None) and (valence is None):
            # for testing without emotion prediction
            angle = np.random.rand() * 360
        else:
            angle = cb. get_emotion_angle(arousal, valence)
            angle = (angle + 180) % 360
        if 30 <= angle <= 150:
            # in more energetic environment we have more std to express dynamics
            std = 0.6
        else:
            std = 0.3
        c = cb.get_emotion_color_by_angle(angle) + np.random.normal((0, 0, 0), scale=std)
        self.color = cb.to_hex_color(c)


def get_on_rows(matrix_size, feature, maxima):
    if maxima != 0:
        on_lines = int(np.floor(matrix_size[0] * feature / maxima))
    else:
        on_lines = 0
    return on_lines
