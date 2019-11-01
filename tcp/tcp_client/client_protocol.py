# from dotstar import Adafruit_DotStar

from twisted.python import log
from twisted.internet.protocol import Protocol, connectionDone

from twisted.protocols.basic import LineReceiver

class EchoClientProtocol(Protocol):
    def dataReceived(self, data):
        log.msg('Data received {}'.format(data))
        self.transport.loseConnection()

    def connectionMade(self):
        data = 'Hello, Server!'
        self.transport.write(data.encode())
        log.msg('Data send {}'.format(data))

    def connectionLost(self, reason=connectionDone):
        log.msg('Lost Connection due to {}'.format(reason.getErrorMessage()))

class RandomReceiverProtocol(LineReceiver):
    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))

    def lineReceived(self, line):
        if  line.isdigit():
            log.msg('Random integer {}'.format(line))
        else:
            log.msg('Server Requests amount of integers, we want 5')
            self.sendLine(b'5')


    def connectionLost(self, reason=connectionDone):
        print('Connection lost from {}'.format(self.transport.getPeer()))

class ReceiveLedProtocol(LineReceiver):
    def __init__(self):
        self.strip = led_strip(300)
        super().__init__(LineReceiver)

    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))
        self.sendLine(b'How many random integers do you want?')

    def lineReceived(self, line):
        num_col = line.strip().split()
        log.msg('Client requested color {} on {}th Pixel'.format(num_col[0], num_col[1]))

        self.strip.setPixel(num_col[0], num_col[1])

    def connectionLost(self, reason=connectionDone):
        print('Connection lost from {}'.format(self.transport.getPeer()))


# def led_strip(numpixels):
#     strip = Adafruit_DotStar(numpixels)
#     strip.begin()
#     return strip