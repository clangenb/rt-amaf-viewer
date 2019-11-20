import numpy as np
import time
import random
from threading import Thread
from queue import Queue

import abc

import visualizer.color.utils as cb

red = cb.to_hex_color(cb.get_emotion_color_by_angle(120))
green = cb.to_hex_color(cb.get_emotion_color_by_angle(0))
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))


class BaseObject:
    def __init__(self, visualizer, coordinate, led_strip, color=red):
        self.vis = visualizer
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.color = color

        self.strip = led_strip
        self.directions = ['up', 'down', 'left', 'right']

        self.current_pixels = self.object_to_pixels()

        self.energy_mean = 1
        self.decay = 0.95

        self.movements = Queue()
        self._t = Thread(target=self.movements_runner)
        self._t.daemon = True
        self._t.start()

    def movements_runner(self):
        while True:
            if not self.movements.empty():
                # print('Movements queue size', self.movements.qsize())
                items = self.movements.get()
                move = items[0]
                args = items[1:]
                move(*args)
            else:
                time.sleep(0.005)

    def bounce_around(self, flux, flux_max, flush=True):
        amount = self.get_bounce_amount(flux, flux_max)

        attack = 0.01
        release = 0.015
        if self.movements.qsize() < 10:
            direction = random.choice(self.directions)
            if direction == 'up':
                self.movements.put((self.move_up, amount, attack))
                self.movements.put((self.move_down, amount, release, True))
            elif direction == 'down':
                self.movements.put((self.move_down, amount, attack))
                self.movements.put((self.move_up, amount, release, True))
            elif direction == 'left':
                self.movements.put((self.move_left, amount, attack))
                self.movements.put((self.move_right, amount, release, True))
            else:
                self.movements.put((self.move_right, amount, attack))
                self.movements.put((self.move_left, amount, release, True))

    def random_move(self, flux, flux_max, flush=True):
        amount = self.get_bounce_amount(flux, flux_max)
        attack = 0.01

        if self.movements.qsize() < 10:
            direction = random.choice(self.directions)
            if direction == 'up':
                self.movements.put((self.move_up, amount, attack, flush))
            elif direction == 'down':
                self.movements.put((self.move_down, amount, attack, flush))
            elif direction == 'left':
                self.movements.put((self.move_left, amount, attack,flush))
            else:
                self.movements.put((self.move_right, amount, attack, flush))

    def pulsate(self, flux, flux_max):
        amount = self.get_pulse_amount(flux, flux_max)
        attack = 0.01
        release = 0.015
        if self.movements.qsize() < 10:
            self.movements.put((self.grow, amount, attack))
            self.movements.put((self.shrink, amount, release))

    def adjust_size(self, energy, energy_max):
        self.energy_mean = self.update_mean(self.energy_mean, energy, self.decay)
        size_factor = self.size_factor(energy)
        attack = 0.01

        print('size factor: {}'.format(size_factor))
        if self.movements.qsize() < 10:
            if energy > self.energy_mean:
                self.movements.put((self.grow, size_factor, attack))
            else:
                self.movements.put((self.shrink, size_factor, attack))

    @abc.abstractmethod
    def get_pulse_amount(self, flux, maxima):
        """ Get pulse amount"""

    def get_bounce_amount(self, feature, maxima):
        if maxima != 0:
            amount = int(np.floor(self.vis.matrix_size[0] / 2.5 * feature / maxima))
        else:
            amount = 0
        return amount

    def get_object_pixels(self):
        return self.object_to_pixels()

    def redraw(self, old_pixels=None, flush=False):
        for i in self.current_pixels:
            self.strip.setPixelColor(i, self.color)
        if flush and (old_pixels is not None):
            self.vis.draw_pixels(old_pixels)

    def size_factor(self, energy):
        if energy > 0:
            return int(np.ceil((energy / self.energy_mean) / 2.5))
        else:
            return 0

    def update_color(self, curr_color):
        c = cb.get_complementary_color(curr_color)
        self.color = cb.to_hex_color(c)
        self.redraw()

    def move_up(self, amount, pause_interval=0.001, flush=False):
        for _ in range(amount):
            self.y += 1
            old_pixels = self.update_pixels()
            self.redraw(old_pixels, flush)
            time.sleep(pause_interval)

    def move_down(self, amount, pause_interval=0.001, flush=False):
        for _ in range(amount):
            self.y -= 1
            old_pixels = self.update_pixels()
            self.redraw(old_pixels, flush)
            time.sleep(pause_interval)

    def move_left(self, amount, pause_interval=0.001, flush=False):
        for _ in range(amount):
            self.x -= 1
            old_pixels = self.update_pixels()
            self.redraw(old_pixels, flush)
            time.sleep(pause_interval)

    def move_right(self, amount, pause_interval=0.001, flush=False):
        for _ in range(amount):
            self.x += 1
            old_pixels = self.update_pixels()
            self.redraw(old_pixels, flush)
            time.sleep(pause_interval)

    def update_pixels(self):
        new_pixels = self.object_to_pixels()
        old_pixels = set(new_pixels).difference(set(self.current_pixels))
        self.current_pixels = new_pixels
        return old_pixels

    def update_mean(self, mean_old, x_n, decay):
        return decay * mean_old + (1 - decay) * x_n

    @abc.abstractmethod
    def object_to_pixels(self):
        """ Get the currently occupied pixels"""

    @abc.abstractmethod
    def grow(self, amount, sleep_interval=0.001):
        """ Increase object size """

    @abc.abstractmethod
    def shrink(self, amount, sleep_interval=0.001, flush=True):
        """ Reduce object size """
