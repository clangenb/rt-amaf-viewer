from twisted.python import log
from twisted.internet.protocol import connectionDone
from twisted.protocols.basic import LineReceiver

from tcp.tcp_server.producer import Producer

class ServeProducerProtocol(LineReceiver):
    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))
        self.sendLine(b'Connection Accepted?')

        producer = Producer(self)
        self.transport.registerProducer(producer, True)
        producer.resumeProducing()

    def lineReceived(self, line):
        msg = line.strip()
        log.msg('Received Msg: '.format(msg))

    def connectionLost(self, reason=connectionDone):
        print('Connection lost from {}'.format(self.transport.getPeer()))