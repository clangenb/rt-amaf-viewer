import numpy as np

import visualizer.color.utils as cb
import visualizer.matrix.pixel_funcs as pxf

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

    def update_color(self, color):
        c = cb.get_complementary_color(color)
        self.color = cb.to_hex_color(c)
        self.redraw()

    def redraw(self):
        for px in self.on_pixels:
            self.strip.setPixelColor(px, self.color)


def get_on_rows(matrix_size, feature, maxima):
    if maxima != 0:
        on_lines = int(np.floor(matrix_size[0] * feature / maxima))
    else:
        on_lines = 0
    return on_lines
