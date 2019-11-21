class TcpStrip:
    def __init__(self, protocol):
        self._proto = protocol

    def setPixelColor(self, i, color):
        # print("Pixel no{}, Pixel Color {}".format(i, color))
        self._proto.sendLine("{},{}".format(i, color).encode("ascii"))

    def show(self):
        self._proto.sendLine("show".encode("ascii"))


class LazyTcpStrip:
    def __init__(self, protocol):
        self._proto = protocol
        self._updated_pixels = {}

    def setPixelColor(self, i, color):
        self._updated_pixels[i] = color

    def show(self):
        if len(self._updated_pixels) > 1:
            leds_to_update = "#"
            for i, color in self._updated_pixels.items():
                leds_to_update += "{},{}#".format(i, color)

            self._proto.sendLine(leds_to_update.encode("ascii"))
