from dotstar import Adafruit_DotStar

from twisted.python import log
from twisted.internet.protocol import connectionDone

from twisted.protocols.basic import LineReceiver

class LedReceiverProtocol(LineReceiver):
    def __init__(self):
        self.strip = led_strip(300)
        super().__init__()

    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))
        self.sendLine(b'How many random integers do you want?')

    def lineReceived(self, line):
        num_col = line.decode("ascii").strip(",").split(",")
        if len(num_col) > 1:
            # log.msg('Set Pixel no {} to Color {}'.format(int(num_col[0]), int(num_col[1])))
            self.strip.setPixelColor(int(num_col[0]), int(num_col[1]))
        else:
            self.strip.show()


    def connectionLost(self, reason=connectionDone):
        print('Connection lost from {}'.format(self.transport.getPeer()))


def led_strip(numpixels):
    strip = Adafruit_DotStar(numpixels)
    strip.begin()
    return strip