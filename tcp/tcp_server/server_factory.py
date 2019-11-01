from twisted.internet.protocol import ServerFactory

from tcp.tcp_server.server_protocol import EchoServerProtocol, ServeRandomProtocol, ReceiveLedProtocol


class EchoServerFactory(ServerFactory):
    def buildProtocol(self, addr):
        return EchoServerProtocol()

class RandomServerFactory(ServerFactory):
    def buildProtocol(self, addr):
        return ServeRandomProtocol()

class LedServerFactory(ServerFactory):
    def buildProtocol(self, addr):
        return ReceiveLedProtocol()