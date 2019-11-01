from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import numpy as np


def get_emotion_color_by_angle(angle, deg=True):
    """ Generate RGB color triplet based on angle in emotion plane"""
    if deg:
        return hsv_to_rgb((angle/360, 1, 1))
    else:
        return hsv_to_rgb((angle/2*np.pi, 1, 1))


def get_emotion_color(arousal, valence):
    angle = get_emotion_angle(arousal, valence, deg=True)
    hue = angle / 360

    if (arousal < 0) and (valence > 0):
        v = 0.95 + arousal / 2
        s = 0.95 - valence / 1.5
    elif (arousal < 0) and (valence < 0):
        v = 0.9 + arousal
        s = 1
    elif (arousal > 0) and (valence < 0):
        v = 0.7 + arousal / 2
        s = 1
    else:
        v = 0.9 + arousal
        s = 1

    h, s, v = np.clip([hue, s, v], a_min=0, a_max=1).flatten()
    return hsv_to_rgb((h, s, v))


def get_emotion_std(arousal, valence):
    """ Define a visually appealing std for the color matrix depending on the detected emotion. """
    if arousal < 0:
        std = 0.09 + abs(arousal) / 2
    elif valence > 0:
        std = 0.17 + arousal / 2
    else:
        std = 0.22 + arousal / 1.5
    return std


def get_complementary_color(color):
    c = np.reshape(color, [-1])
    h, s, v = rgb_to_hsv(c)

    h_c = (h + 0.5) % 1

    cc = np.clip([h_c, s, v], a_min=0, a_max=1).flatten()
    return hsv_to_rgb(cc)


def get_emotion_angle(arousal, valence, deg=True):
    angle = np.angle(arousal * 1j + valence, deg=True)
    angle = renormalize_angle(angle)

    if deg:
        return angle
    else:
        return angle / 360 * 2*np.pi


def renormalize_angle(angle, deg=True):
    """
    Transform the angle obtained from np.angle to [0, 360]
    :param angle: [-180, 180]
    :param def: if true angle is in degree, else radians
    :return: [0, 360]
    """
    if deg:
        if 0 < angle <= 180:
            return angle
        elif -180 < angle <= 0:
            return 360 + angle
    else:
        if 0 < angle <= np.pi:
            return angle
        elif -np.pi < angle <= 0:
            return 2*np.pi + angle


def gaussian_color_matrix(color, std, size):
    """ Generates a matrix containing color triplets """
    base_color = color
    color_matrix = [base_color for _ in range(size[0]) for _ in range(size[1])]
    color_matrix = np.reshape(color_matrix, (size[0], size[1], 3))
    color_matrix = np.ndarray.astype(color_matrix, dtype=float, copy=False)
    return np.random.normal(color_matrix, std)


def gaussian_color_matrix_rand_hvs_only(color, std, size):
    base_color = rgb_to_hsv(color)[0]
    color_matrix = [base_color for _ in range(size[0]) for _ in range(size[1])]
    color_matrix = np.asarray(color_matrix, dtype=float)

    m = [hsv_to_rgb([c, 1, 1]) for c in np.random.normal(color_matrix, std).flatten()]
    return np.reshape(m, (size[0], size[1], 3))


def get_object_color(base_color, std):
    return base_color + np.random.normal((0, 0, 0), scale=std)


def to_hex_array(color_array):
    """ Transforms RGB color array to hex color array"""
    c_array = color_array.reshape([-1, 3])
    hex_array = []
    for row in c_array:
        hex_str = '0x%02x%02x%02x' % (clamp(row[0]), clamp(row[1]), clamp(row[2]))
        hex_array.append(int(hex_str, 16))
    # print(hex_array)
    # at the moment return just a list of colors for testing
    return hex_array


def to_hex_color(color, is_rgb=True):
    if is_rgb:
        hex_str = '0x%02x%02x%02x' % (clamp(color[0]), clamp(color[1]), clamp(color[2]))
    else:
        c_rgb = hsv_to_rgb(color)
        hex_str = '0x%02x%02x%02x' % (clamp(c_rgb[0]), clamp(c_rgb[1]), clamp(c_rgb[2]))

    return int(hex_str, 16)


def clamp(color_rgb):
    return max(0, min(int(255*color_rgb), 255))
