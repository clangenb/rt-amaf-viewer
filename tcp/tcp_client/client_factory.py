from twisted.python import log
from twisted.internet.protocol import ClientFactory

from tcp.tcp_client.client_protocol import EchoClientProtocol, RandomReceiverProtocol

class EchoClientFactory(ClientFactory):
    def startedConnecting(self, connector):
        log.msg('Connecting...')

    def buildProtocol(self, addr):
        log.msg('Connected')
        return EchoClientProtocol()

    def clientConnectionLost(self, connector, reason):
        log.msg('Lost connection. Reason: {}'.format(reason.getErrorMessage()))

    def clientConnectionFailed(self, connector, reason):
        log.msg('Connection failed. Reason {}'.format(reason.getErrorMessage()))

class RandomReceiverFactory(ClientFactory):
    def startedConnecting(self, connector):
        log.msg('Connecting...')

    def buildProtocol(self, addr):
        log.msg('Connected')
        return RandomReceiverProtocol()

    def clientConnectionLost(self, connector, reason):
        log.msg('Lost connection. Reason: {}'.format(reason.getErrorMessage()))

    def clientConnectionFailed(self, connector, reason):
        log.msg('Connection failed. Reason {}'.format(reason.getErrorMessage()))