from dotstar import Adafruit_DotStar

class PhysicalStrip:
    def __init__(self, numpixels):
        self._strip = Adafruit_DotStar(numpixels)
        self._strip.begin()

    def setPixelColor(self, i, color):
        self._strip.setPixelColor(i, color)

    def show(self):
        self._strip.show()