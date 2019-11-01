import time

import numpy as np
from dotstar import Adafruit_DotStar

import visualizer.color_base_functions as cb
from visualizer.patterns.rectangle_object import Rectangle

smile_entropy = 'pcm_fftMag_spectralEntropy_sma'

red = cb.to_hex_color(cb.get_emotion_color_by_angle(120))
green = cb.to_hex_color(cb.get_emotion_color_by_angle(0))
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))
violett = cb.to_hex_color(cb.get_emotion_color_by_angle(180))

numpixels = 300


def led_strip():
    strip = Adafruit_DotStar(numpixels)
    strip.begin()
    return strip


def show_all(strip, color):
    for i in range(numpixels):
        strip.setPixelColor(i, color)

    strip.show()


def show_blue(strip):
    show_all(strip, blue)


def show_green(strip):
    show_all(strip, green)


def show_red(strip):
    show_all(strip, red)


def show_violett(strip):
    show_all(strip, violett)
