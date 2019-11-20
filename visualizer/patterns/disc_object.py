import numpy as np
import time

import visualizer.pixel_funcs as pf
import visualizer.color.utils as cb
from visualizer.patterns.base_object import BaseObject

red = cb.to_hex_color(cb.get_emotion_color_by_angle(120))
green = cb.to_hex_color(cb.get_emotion_color_by_angle(0))
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))


class Disc(BaseObject):
    def __init__(self, visualizer, coordinate, radius, led_strip, color=red):
        self.matrix_center = (10, 8)
        self.radius = radius

        super().__init__(visualizer, coordinate, led_strip, color)

    def get_pulse_amount(self, flux, maxima):
        if maxima != 0:
            amount = int(np.floor(self.radius * 2 * flux / maxima))
        else:
            amount = 0
        return amount

    def get_bounce_amount(self, feature, maxima):
        if maxima != 0:
            amount = int(np.floor(self.vis.matrix_size[0] / 2.5 * feature / maxima))
        else:
            amount = 0
        return amount

    def grow(self, amount, sleep_interval=0.001):
        self.radius += np.ceil(amount / 2)

        self.redraw()
        time.sleep(sleep_interval)

    def shrink(self, amount, sleep_interval=0.001, flush=True):

        if self.radius > np.ceil(amount / 2):
            self.radius -= np.ceil(amount / 2)

        old_pixels = self.update_pixels()

        self.redraw(old_pixels, flush)
        time.sleep(sleep_interval)

    def object_to_pixels(self):
        pixels = []

        for x in range(self.vis.matrix_size[1]):
            for y in range(self.vis.matrix_size[0]):
                if int((x - self.x)**2 + (y - self.y)**2 - 0.5) < self.radius**2:
                    pixels.append(pf.to_pixel_no((x, y)))

        return pixels
