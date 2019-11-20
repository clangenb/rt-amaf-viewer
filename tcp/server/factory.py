from twisted.internet.protocol import ServerFactory

from tcp.server.protocol import ServeProducerProtocol

class ProducerFactory(ServerFactory):
    def buildProtocol(self, addr):
        return ServeProducerProtocol()
