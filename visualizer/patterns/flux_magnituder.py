import numpy as np

black = 0x000000


class FluxMagnituder:
    def __init__(self, led_strip):
        self.matrix_size = (15, 20)
        self.curr_on_lines = 0
        self.strip = led_strip

    def update_on_pixels(self, hex_array):
        for i in range(0, self.curr_on_lines):
            j = 0
            for color in hex_array[i * self.matrix_size[1]:(i+1)*self.matrix_size[1]]:
                self.strip.setPixelColor(i * self.matrix_size[1] + j, color)
                j += 1
        self.strip.show()

    def show(self, hex_array, flux, flux_maxima):
        next_lines_on = get_on_ratio(self.matrix_size, flux, flux_maxima)
        self._draw_attack(hex_array, self.curr_on_lines, next_lines_on)
        self.curr_on_lines = next_lines_on

    def _draw_attack(self, hex_array, curr_on_lines, next_lines_on):
        if curr_on_lines > next_lines_on:
            for i in range(curr_on_lines, next_lines_on, -1):
                for j in range(self.matrix_size[1]):
                    self.strip.setPixelColor(i*self.matrix_size[1] + j, black)
                self.strip.show()
        else:
            for i in range(curr_on_lines, next_lines_on):
                j = 0
                for color in hex_array[i*self.matrix_size[1]:(i + 1)*self.matrix_size[1]]:
                    self.strip.setPixelColor(i * self.matrix_size[1] + j, color)
                    j += 1
                self.strip.show()


def get_on_ratio(matrix_size, feature, maxima):
    if maxima != 0:
        on_lines = int(np.floor(matrix_size[0] * feature / maxima))
    else:
        on_lines = 0
    return on_lines
