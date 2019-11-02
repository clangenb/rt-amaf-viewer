from twisted.internet.protocol import ServerFactory

from tcp.tcp_server.server_protocol import ServeProducerProtocol

class ProducerFactory(ServerFactory):
    def buildProtocol(self, addr):
        return ServeProducerProtocol()
