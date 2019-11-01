import configparser

import numpy as np
from dotstar import Adafruit_DotStar

import visualizer.color_base_functions as cb
import visualizer.color_base_functions as cbf
from utility.time_quantizer import TimeQuantizer
from visualizer.backgrounds.background import Backgrounder
from visualizer.patterns.disc_object import Disc
from visualizer.patterns.flux_magnituder import FluxMagnituder
from visualizer.patterns.rectangle_object import Rectangle
from visualizer.patterns.spectrum import CoefficientShower

#################################################################################
# Feature names of the features extracted from openSMILE
#################################################################################

config_file = "visualizer_conf.ini"

smile_entropy = 'pcm_fftMag_spectralEntropy_sma'
smile_centroid = 'pcm_fftMag_spectralCentroid_sma'
smile_flux = 'pcm_fftMag_spectralFlux_sma'
smile_hnr = 'logHNR_sma'
smile_harmonicity = 'pcm_fftMag_spectralHarmonicity_sma'
smile_rms = 'pcm_RMSenergy_sma'
smile_delta_rms = 'pcm_RMSenergy_sma_de'
smile_band250_650 = 'pcm_fftMag_fband250-650_sma'
smile_rolloff = 'pcm_fftMag_spectralRollOff75.0_sma'

smile_mfccs = ['mfcc_sma[{}]'.format(i) for i in range(1, 14)]
smile_rastas = ['audSpec_Rfilt_sma[{}]'.format(i) for i in range(26)]


black = 0x000000
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))
yellow = cb.to_hex_color(cb.get_emotion_color_by_angle(60))


visualizer_types = []


class Visualizer:
    def __init__(self, feature_list, std, type='rasta_shower'):
        self.matrix_size = (15, 20)
        self.backgrounder = Backgrounder(std, self.matrix_size)
        self.curr_off_pixels = []
        self.curr_object_pixels = []

        self.curr_color = cbf.get_emotion_color_by_angle(60)
        self.hex_array = cbf.to_hex_array(cbf.gaussian_color_matrix(self.curr_color, std=std, size=self.matrix_size))
        self.enabled_features = {}

        self.timer = TimeQuantizer()

        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform = str
        config.read(config_file)

        self.numpixels = 300
        self.strip = led_strip(numpixels=self.numpixels)

        self.magnituder = FluxMagnituder(self.strip)
        self.rasta_shower = CoefficientShower(self.strip, len(smile_rastas))
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

        for key in config['features']:
            if int(config['features'][key]) == 1 and key in feature_list:
                self.enabled_features[key] = feature_list.index(key)

        # for mf in smile_mfccs:
        #     if mf in feature_list:
        #         self.enabled_features[mf] = feature_list.index(mf)

        # for ra in smile_rastas:
        #     if ra in feature_list:
        #         self.enabled_features[ra] = feature_list.index(ra)

        print('Enabled Features: ', self.enabled_features)
        self.feature_maxima = np.zeros(len(feature_list))

    def update_visuals(self, llds):
        self.timer.reset()

        self.feature_maxima = np.maximum(self.feature_maxima, llds)
        self.feature_maxima = self.feature_maxima * 0.999
        # rastas = self._get_rastas(llds)

        if smile_flux in self.enabled_features:
            flux, centroid, rms, entropy, flux_max, energy_max, \
            energy_delta, spec_rolloff, hnr = self._get_features(llds)

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

    def _get_features(self, llds):
        flux = llds[self.enabled_features[smile_flux]]
        centroid = llds[self.enabled_features[smile_centroid]]
        rms = llds[self.enabled_features[smile_rms]]
        entropy = llds[self.enabled_features[smile_entropy]]
        flux_max = self.feature_maxima[self.enabled_features[smile_flux]]
        energy_max = self.feature_maxima[self.enabled_features[smile_rms]]
        energy_delta = llds[self.enabled_features[smile_delta_rms]]
        spect_rolloff = llds[self.enabled_features[smile_rolloff]]
        # hnr = llds[self.enabled_features[smile_hnr]]
        spect_harm = llds[self.enabled_features[smile_harmonicity]]

        return flux, centroid, rms, entropy, flux_max, energy_max, energy_delta, spect_rolloff, spect_harm

    def _get_mfccs(self, llds):
        mfccs = [llds[self.enabled_features[mf]] for mf in smile_mfccs]
        return mfccs

    def _get_rastas(self, llds):
        rastas = [llds[self.enabled_features[ra]] for ra in smile_rastas]
        return rastas

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


def led_strip(numpixels):
    strip = Adafruit_DotStar(numpixels)
    strip.begin()
    return strip
