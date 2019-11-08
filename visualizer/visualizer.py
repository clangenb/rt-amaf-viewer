import configparser
import numpy as np

import visualizer.color_base_functions as cbf
from utility.time_quantizer import TimeQuantizer
from visualizer.backgrounds.background import Backgrounder
from visualizer.patterns.disc_object import Disc
from visualizer.patterns.flux_magnituder import FluxMagnituder
from visualizer.patterns.rectangle_object import Rectangle
from visualizer.patterns.spectrum import CoefficientShower
from visualizer.smile_features import HLDs, EnabledFeatures

black = 0x000000
blue = cbf.to_hex_color(cbf.get_emotion_color_by_angle(220))
yellow = cbf.to_hex_color(cbf.get_emotion_color_by_angle(60))

config_file = "visualizer_conf.ini"

visualizer_types = []


class Visualizer:
    def __init__(self, feature_list, std, tcp_protocol, type='rasta_shower'):
        self.matrix_size = (15, 20)
        self.backgrounder = Backgrounder(std, self.matrix_size)
        self.curr_off_pixels = []
        self.curr_object_pixels = []

        self.curr_color = cbf.get_emotion_color_by_angle(60)
        self.hex_array = cbf.to_hex_array(cbf.gaussian_color_matrix(self.curr_color, std=std, size=self.matrix_size))

        self.timer = TimeQuantizer()

        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform = str
        config.read(config_file)

        self.numpixels = 300
        self.strip = tcp_strip(tcp_protocol)

        self.magnituder = FluxMagnituder(self.strip)
        self.rasta_shower = CoefficientShower(self.strip, len(HLDs.rastas))
        # rec1 = Rectangle(visualizer=self, coordinate=(4, 4), size=(4, 4), led_strip=self.strip)
        # rec2 = Rectangle(visualizer=self, coordinate=(9, 4), size=(2, 2), led_strip=self.strip, color=yellow)
        # rec3 = Rectangle(visualizer=self, coordinate=(14, 4), size=(2, 2), led_strip=self.strip, color=yellow)
        # rec4 = Rectangle(visualizer=self, coordinate=(4, 9), size=(2, 2), led_strip=self.strip, color=yellow)
        # rec5 = Rectangle(visualizer=self, coordinate=(3, 9), size=(2, 2), led_strip=self.strip, color=yellow)
        # rec6 = Rectangle(visualizer=self, coordinate=(14, 9), size=(2, 2), led_strip=self.strip, color=yellow)
        rec6 = Rectangle(visualizer=self, coordinate=(10, 6), size=(3, 3), led_strip=self.strip, color=yellow)

        circle = Disc(visualizer=self, coordinate=(10, 8), radius=1.5, led_strip=self.strip)
        # self.objects = [rec1, rec2, rec3, rec4, rec5, rec6]
        self.objects = [rec6]

        self.enabled_features = EnabledFeatures(config, feature_list)

        # for mf in HLD.smile_mfccs:
        #     if mf in feature_list:
        #         self.enabled_features[mf] = feature_list.index(mf)

        # for ra in HLD.smile_rastas:
        #     if ra in feature_list:
        #         self.enabled_features[ra] = feature_list.index(ra)

        print('Enabled Features: ', self.enabled_features)
        self.feature_maxima = np.zeros(len(feature_list))

    def update_visuals(self, llds):
        self.timer.reset()

        self.feature_maxima = np.maximum(self.feature_maxima, llds)
        self.feature_maxima = self.feature_maxima * 0.999
        # rastas = self._get_rastas(llds)

        if HLDs.flux in self.enabled_features:
            flux, centroid, rms, entropy, flux_max, energy_max, \
            energy_delta, spec_rolloff, hnr = self.enabled_features.get_features(self.feature_maxima, llds)

            # self.rasta_shower.show(rastas)
            self.update_palette(centroid, rms, hnr)
            # for o in self.objects:
            #     o.redraw()
            if (flux > flux_max / 3) and (energy_delta > 0.007):
                # self.circle.bounce_around(flux, flux_max)
                for rec in self.objects:
                    rec.random_move(flux, flux_max, flush=True)
            #
            # # self.magnituder.show(self.hex_array, flux, flux_max)
            # print('updating visuals time: ', self.timer.measure_total())

        self.strip.show()

    def update_base_color(self, arousal, valence):
        self.curr_color = cbf.get_emotion_color(arousal, valence)
        self.backgrounder.update_gaussian_mask(arousal, valence)
        for o in self.objects:
            o.update_color(self.curr_color)

    def update_palette(self, centroid, rms, entropy):
        self.hex_array, off_pixels = self.backgrounder.modulate_color(self.curr_color, centroid, rms, entropy)
        self.draw_whole_background(off_pixels)
        self.timer.reset()
        for o in self.objects:
            o.redraw()

    def _draw_all(self, hex_array):
        i = 0
        for color in hex_array:
            self.strip.setPixelColor(i, color)
            i += 1

        self.strip.show()

    def draw_whole_background(self, off_pixels=None):
        non_background_pixels = []
        object_pixels = []
        for o in self.objects:
            object_pixels += o.get_object_pixels()
        if off_pixels is not None:
            for i in off_pixels:
                if i not in object_pixels:
                    self.strip.setPixelColor(i, black)
                # else:
                #     for o in self.objects:
                #         if i in o.get_object_pixels():
                #             self.strip.setPixelColor(i, o.color)

            non_background_pixels += off_pixels
            non_background_pixels += object_pixels
            self.curr_off_pixels = off_pixels
            self.curr_object_pixels = object_pixels

        i = 0
        if len(non_background_pixels) > 150:
            background_pixels = set(range(300)).difference(set(non_background_pixels))
            for color in self.hex_array:
                if i in background_pixels:
                    self.strip.setPixelColor(i, color)
                i += 1
        else:
            for color in self.hex_array:
                if i not in non_background_pixels:
                    self.strip.setPixelColor(i, color)
                i += 1

    def draw_pixels(self, pixels):
        i = 0
        object_pixels = []
        for o in self.objects:
            object_pixels += o.get_object_pixels()
        for color in self.hex_array:
            if i in pixels:
                if i in self.curr_off_pixels:
                    self.strip.setPixelColor(i, black)
                elif i not in self.curr_object_pixels:
                    self.strip.setPixelColor(i, color)
                else:
                    for o in self.objects:
                        if i in o.get_object_pixels():
                            self.strip.setPixelColor(i, o.color)
            i += 1
        self.strip.show()



class tcp_strip:
    def __init__(self, protocol):
        self._proto = protocol

    def show(self):
        self._proto.sendLine("show".encode("ascii"))

    def setPixelColor(self, i, color):
        # print("Pixel no{}, Pixel Color {}".format(i, color))
        self._proto.sendLine("{},{}".format(i, color).encode("ascii"))

