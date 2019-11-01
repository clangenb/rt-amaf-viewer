from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def fade_iterator(start_color, end_color, steps):
    """ start_-, end_color in RGB floats element of [0,1] """
    for i in range(steps):
        yield start_color + (end_color - start_color)/steps * i


def move_image_upwards(color_array):
    colors = color_array.copy()
    colors[1:] = color_array[0:-1]
    colors[0] = color_array[-1]
    return colors


def brightness_modulation(color, energy):
    hsv = rgb_to_hsv(color)
    hsv[:, :, 2] = energy
    return hsv_to_rgb(hsv)


def brightness_iterator(color, start_energy, end_energy, steps):
    for i in range(steps):
        energy = start_energy + (end_energy - start_energy) / steps * i
        yield brightness_modulation(color, energy)
