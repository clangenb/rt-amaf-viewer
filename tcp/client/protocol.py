from twisted.python import log
from twisted.internet.protocol import connectionDone

from twisted.protocols.basic import LineReceiver


class LedReceiverProtocol(LineReceiver):
    def __init__(self, led_strip):
        self.strip = led_strip
        super().__init__()

    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))
        self.sendLine(b'How many random integers do you want?')

    def lineReceived(self, line):
        # pixel_tuples should be ["i, color"]
        line_dec = line.decode("ascii")
        if line_dec[0] == "#":
            pixel_tuples = line.decode("ascii").strip("#").split("#")
            # print("Pixel tuples", pixel_tuples)
            for pixel_tuple in pixel_tuples:
                pixel, color = pixel_tuple.split(",")
                # print('Set Pixel no {} to Color {}'.format(int(pixel), int(color)))
                self.strip.setPixelColor(int(pixel), int(color))
        else:
            self.strip.show()

    def connectionLost(self, reason=connectionDone):
        print('Connection lost from {}'.format(self.transport.getPeer()))
