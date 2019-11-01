import numpy as np
import time

import visualizer.pixel_funcs as pf
import visualizer.color_base_functions as cb
from visualizer.patterns.base_object import BaseObject

red = cb.to_hex_color(cb.get_emotion_color_by_angle(120))
green = cb.to_hex_color(cb.get_emotion_color_by_angle(0))
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))


class Rectangle(BaseObject):
    def __init__(self, visualizer, coordinate, size, led_strip, color=red):

        self.width = size[0]
        self.height = size[1]
        self.max = 13

        super().__init__(visualizer, coordinate, led_strip, color)

    def get_pulse_amount(self, flux, maxima):
        if maxima != 0:
            amount = int(np.floor(self.width * 2 * flux / maxima))
        else:
            amount = 0
        return amount

    def grow(self, amount, sleep_interval=0.001):
        for _ in range(amount):
            if self.width < self.max and self.height < self.max:
                self.x = (self.x - 1)   # % self.vis.matrix_size[1]
                self.y = (self.y - 1)   # % self.vis.matrix_size[0]
                self.width = (self.width + 1)   # % self.vis.matrix_size[1]
                self.height = (self.height + 1)   # % self.vis.matrix_size[0]
                self.update_pixels()

                self.redraw()
                time.sleep(sleep_interval)

    def rotate(self, rotrad):
        cords = [pf.pixel_to_matrix_cord(p) for p in self.current_pixels]
        new_cords = [pf.rotate(c, (self.x, self.y), rotrad) for c in cords]
        new_pixels = [pf.to_pixel_no(c) for c in new_cords]
        old_pixels = set(new_pixels).difference(set(self.current_pixels))
        self.current_pixels = new_pixels
        self.redraw(old_pixels, True)

    def shrink(self, amount, sleep_interval=0.001, flush=True):
        for _ in range(amount):
            if min(self.width, self.height) > 2:
                # old_pixels = []
                # old_pixels.extend(self.get_lowest_row())
                # old_pixels.extend(self.get_highest_row())
                # old_pixels.extend(self.get_leftmost_column())
                # old_pixels.extend(self.get_rightermost_column())

                self.x = (self.x + 1)   # % self.vis.matrix_size[1]
                self.y = (self.y + 1)   # % self.vis.matrix_size[0]
                self.width -= 1
                self.height -= 1
                old_pixels = self.update_pixels()

                self.redraw(old_pixels, flush)
                time.sleep(sleep_interval)

    def get_lowest_row(self):
        row = []
        for i in range(self.width):
            x = (self.x + i) % 20
            y = self.y % 15
            row.append(pf.to_pixel_no((x, y)))
        return row

    def get_highest_row(self):
        row = []
        for i in range(self.width):
            x = (self.x + i) % 20
            y = (self.y + self.height - 1) % 15
            row.append(pf.to_pixel_no((x, y)))
        return row

    def get_rightermost_column(self):
        col = []
        for i in range(self.height):
            x = (self.x + self.width - 1) % 20
            y = (self.y + i) % 15
            col.append(pf.to_pixel_no((x, y)))
        return col

    def get_leftmost_column(self):
        col = []
        for i in range(self.height):
            x = self.x % 20
            y = (self.y + i) % 15
            col.append(pf.to_pixel_no((x, y)))
        return col

    def object_to_pixels(self):
        pixels = []
        for h in range(self.height):
            for w in range(self.width):
                x = (self.x + w) % self.vis.matrix_size[1]
                y = (self.y + h) % self.vis.matrix_size[0]
                pixels.append(pf.to_pixel_no((x, y)))

        return pixels
