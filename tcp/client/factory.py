from twisted.python import log
from twisted.internet.protocol import ClientFactory

from tcp.client.protocol import LedReceiverProtocol
from visualizer.matrix.leds.physical_strip import PhysicalStrip


class LedReceiverFactory(ClientFactory):
    def startedConnecting(self, connector):
        log.msg('Connecting...')

    def buildProtocol(self, addr):
        log.msg('Connected')
        strip = PhysicalStrip(300)
        return LedReceiverProtocol(strip)

    def clientConnectionLost(self, connector, reason):
        log.msg('Lost connection. Reason: {}'.format(reason.getErrorMessage()))

    def clientConnectionFailed(self, connector, reason):
        log.msg('Connection failed. Reason {}'.format(reason.getErrorMessage()))