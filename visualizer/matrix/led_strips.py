class TcpStrip:
    def __init__(self, protocol):
        self._proto = protocol

    def show(self):
        self._proto.sendLine("show".encode("ascii"))

    def setPixelColor(self, i, color):
        # print("Pixel no{}, Pixel Color {}".format(i, color))
        self._proto.sendLine("{},{}".format(i, color).encode("ascii"))