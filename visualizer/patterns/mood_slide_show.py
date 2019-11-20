import visualizer.color.color_base_functions as cb
from dotstar import Adafruit_DotStar

black = 0x000000
blue = cb.to_hex_color(cb.get_emotion_color_by_angle(220))


class MoodSlider:
    def __init__(self):
        self.matrix_size = (15,20)
        self.color_array = [black for _ in range(self.matrix_size[0])]
        self.curr_color = black
        self.strip = led_strip(300)

    def slide(self, arousal=None, valence=None):

        self.color_array[1:] = self.color_array[0:-1]
        if (arousal is None) and (valence is None):
            self.color_array[0] = self.curr_color
        else:
            self.curr_color = cb.to_hex_color(cb.get_emotion_color(arousal, valence))
            self.color_array[0] = self.curr_color

        self.refresh()


    def refresh(self):
        for i in range(300):
            self.strip.setPixelColor(i, self.color_array[int(i/self.matrix_size[1])])

        self.strip.show()



def led_strip(numpixels):
    strip = Adafruit_DotStar(numpixels)
    strip.begin()
    return strip