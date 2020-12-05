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
    BouncingSquare = "Normal",
    MultiBouncingSquare = "multi"
    Rasta = "rasta"
    Magnituder = "magnituder"


class Visualizer:
    def __init__(self, feature_list, std, led_strip, vis_type=VisualizerTypes.BouncingSquare):
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
        self.rec = Rectangle(visualizer=self, coordinate=(10, 6), size=(3, 3), led_strip=self.strip, color=yellow)
        if self.type == VisualizerTypes.BouncingSquare:
            self.objects = [self.rec]

        self.enabled_features = EnabledFeatures(config, feature_list)
        print('Enabled Features: ', self.enabled_features.list())

        self.feature_maxima = np.zeros(len(feature_list))

        self.rasta_shower = CoefficientShower(self.strip, len(HLDs.rastas))
        if self.type == VisualizerTypes.Rasta:
            self.objects = [self.rasta_shower]

        # if self.type == VisualizerTypes.Magnituder:
        self.magnituder = FluxMagnituder(self.strip)

    def update_visuals(self, llds):
        self.feature_maxima = np.maximum(self.feature_maxima, llds)
        self.feature_maxima = self.feature_maxima * 0.999

        if self.type == VisualizerTypes.Rasta:
            self.update_rastas(llds)

        flux, centroid, rms, entropy, flux_max, energy_max, \
        energy_delta, spec_rolloff, hnr = self.enabled_features.get_features(self.feature_maxima, llds)

        self.update_palette(centroid, rms, hnr)

        if should_trigger_movement(flux, flux_max, energy_delta):
            if self.timer.measure_total() > 3 * 60:
                print('updating visuals')
                self.next_visualizer_type()
                self.timer.reset()

            if self.is_bouncing_squares():
                if self.timer.measure_tick() > 10:
                    self.initiate_switch_bounce(flux, flux_max)
                else:
                    for rec in self.objects:
                        rec.random_move(flux, flux_max, flush=True)

        if self.type == VisualizerTypes.Magnituder:
            self.magnituder.show(self.hex_array, flux, flux_max)
        #
        # # print('updating visuals time: ', self.timer.measure_total())
        #
        self.strip.show()

    def update_rastas(self, llds):
        rastas = self.enabled_features.get_rastas(llds)
        self.rasta_shower.show(rastas)

    def update_base_color(self, arousal, valence):
        self.curr_color = cbf.get_emotion_color(arousal, valence)
        self.backgrounder.update_gaussian_mask(arousal, valence)
        for o in self.objects:
            o.update_color(self.curr_color)

    def update_palette(self, centroid, rms, entropy):
        self.hex_array, self.curr_off_pixels = self.backgrounder.modulate_color(self.curr_color, centroid, rms, entropy)
        # if self.type == VisualizerTypes.BouncingSquare or self.type == VisualizerTypes.MultiBouncingSquare:
        #     self.curr_off_pixels = set(range(300))

        self.curr_object_pixels = self.get_object_pixels()
        self.draw_whole_background(self.curr_object_pixels, self.curr_off_pixels)
        for o in self.objects:
            o.redraw()

    def _draw_all(self, hex_array):
        i = 0
        for color in hex_array:
            self.strip.setPixelColor(i, color)
            i += 1

        self.strip.show()

    def draw_whole_background(self, object_pixels, off_pixels):
        if off_pixels is not None:
            self.set_off_pixels_black(object_pixels, off_pixels)
        non_background_pixels = []
        non_background_pixels += object_pixels
        non_background_pixels += off_pixels
        self.draw_background(non_background_pixels)

    def get_object_pixels(self):
        object_pixels = []
        for o in self.objects:
            object_pixels += o.get_object_pixels()
        return object_pixels

    def draw_background(self, non_background_pixels):
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

    def set_off_pixels_black(self, object_pixels, off_pixels):
        for i in off_pixels:
            if i not in object_pixels:
                self.strip.setPixelColor(i, black)

    def draw_pixels(self, pixels):
        i = 0
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

    def next_visualizer_type(self):
        if self.type == VisualizerTypes.BouncingSquare:
            self.update_visualizer_type(VisualizerTypes.MultiBouncingSquare)
        elif self.type == VisualizerTypes.MultiBouncingSquare:
            self.update_visualizer_type(VisualizerTypes.Magnituder)
        elif self.type == VisualizerTypes.Magnituder:
            self.update_visualizer_type(VisualizerTypes.Rasta)
        else:
            self.update_visualizer_type(VisualizerTypes.BouncingSquare)

    def update_visualizer_type(self, new_type):
        if new_type == VisualizerTypes.BouncingSquare:
            self.type = VisualizerTypes.BouncingSquare
            self.objects = [self.rec]
        elif new_type == VisualizerTypes.MultiBouncingSquare:
            self.type = VisualizerTypes.MultiBouncingSquare
            self.objects = self.get_rectangle_array(self.rec.x, self.rec.y)
        elif new_type == VisualizerTypes.Magnituder:
            self.type = VisualizerTypes.Magnituder
            self.objects = []
        elif new_type == VisualizerTypes.Rasta:
            self.type = VisualizerTypes.Rasta
            self.objects = [self.rasta_shower]
        else:
            self.type = VisualizerTypes.BouncingSquare
            self.objects = [self.rec]

    def initiate_switch_bounce(self, flux, flux_max):
        if self.type == VisualizerTypes.MultiBouncingSquare:
            if self.all_objects_at(self.rec.x, self.rec.y):
                self.update_visualizer_type(VisualizerTypes.BouncingSquare)
                self.timer.set_tick()
            else:
                self.all_objects_approach(self.rec.x, self.rec.y, flux, flux_max)

        else:
            self.update_visualizer_type(VisualizerTypes.MultiBouncingSquare)
            self.timer.set_tick()

    def is_bouncing_squares(self):
        return self.type == VisualizerTypes.BouncingSquare or self.type == VisualizerTypes.MultiBouncingSquare

    def all_objects_at(self, x, y):
        for rec in self.objects:
            if not rec.is_at(x, y):
                return False

        return True

    def all_objects_approach(self, x, y, flux, flux_max):
        [rec.approach(x, y, flux, flux_max) for rec in self.objects]

    def get_rectangle_array(self, x, y):
        return [Rectangle(visualizer=self, coordinate=(x, y), size=(1, 1), led_strip=self.strip, color=yellow)
                for _ in range(20)]


def should_trigger_movement(flux, flux_max, energy_delta):
    return (flux > flux_max / 5) and (energy_delta > 0.001)
