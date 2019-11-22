import configparser
import numpy as np

import visualizer.color.utils as cbf
from utility.time_quantizer import TimeQuantizer
from visualizer.backgrounds.background import Backgrounder
from visualizer.patterns.flux_magnituder import FluxMagnituder
from visualizer.patterns.rectangle_object import Rectangle
from visualizer.patterns.spectrum import CoefficientShower
from visualizer.smile_features import HLDs, EnabledFeatures

black = 0x000000
blue = cbf.to_hex_color(cbf.get_emotion_color_by_angle(220))
yellow = cbf.to_hex_color(cbf.get_emotion_color_by_angle(60))

config_file = "visualizer_conf.ini"

class VisualizerTypes:
    Normal = "Normal"
    Rasta = "rasta"
    Magnituder = "magnituderr"


class Visualizer:
    def __init__(self, feature_list, std, led_strip, vis_type='rasta_shower'):
        self.matrix_size = (15, 20)
        self.backgrounder = Backgrounder(std, self.matrix_size)
        self.curr_off_pixels = []
        self.curr_object_pixels = []
        self.type = vis_type

        self.curr_color = cbf.get_emotion_color_by_angle(60)
        self.hex_array = cbf.to_hex_array(cbf.gaussian_color_matrix(self.curr_color, std=std, size=self.matrix_size))

        self.timer = TimeQuantizer()

        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform = str
        config.read(config_file)

        self.numpixels = 300
        self.strip = led_strip
        rec6 = Rectangle(visualizer=self, coordinate=(10, 6), size=(3, 3), led_strip=self.strip, color=yellow)

        self.objects = [rec6]

        self.enabled_features = EnabledFeatures(config, feature_list)
        print('Enabled Features: ', self.enabled_features.list())

        self.feature_maxima = np.zeros(len(feature_list))

        if self.type == VisualizerTypes.Rasta:
            self.rasta_shower = CoefficientShower(self.strip, len(HLDs.rastas))

        if self.type == VisualizerTypes.Magnituder:
            self.magnituder = FluxMagnituder(self.strip)

    def update_visuals(self, llds):
        self.timer.reset()

        self.feature_maxima = np.maximum(self.feature_maxima, llds)
        self.feature_maxima = self.feature_maxima * 0.999

        if self.type == VisualizerTypes.Rasta:
            self.update_rastas(llds)

        flux, centroid, rms, entropy, flux_max, energy_max, \
            energy_delta, spec_rolloff, hnr = self.enabled_features.get_features(self.feature_maxima, llds)

        self.update_palette(centroid, rms, hnr)
        if (flux > flux_max / 3) and (energy_delta > 0.007):
            for rec in self.objects:
                rec.random_move(flux, flux_max, flush=True)

        if self.type == VisualizerTypes.Magnituder:
            self.magnituder.show(self.hex_array, flux, flux_max)

        # print('updating visuals time: ', self.timer.measure_total())

        self.strip.show()

    def update_rastas(self, llds):
        rastas = self.enabled_features.get_mfccs(llds)
        self.rasta_shower.show(rastas)
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
