from twisted.python import log
from twisted.internet.protocol import Protocol, connectionDone
from twisted.protocols.basic import LineReceiver

from tcp.tcp_server.producer import Producer

class EchoServerProtocol(Protocol):
    def dataReceived(self, data):
        log.msg('Data received {}'.format(data))
        self.transport.write(data)

    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))

    def connectionLost(self, reason=connectionDone):
        log.msg('Lost connection due to {}'.format(reason))


class ServeRandomProtocol(LineReceiver):
    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))
        self.sendLine(b'How many random integers do you want?')

    def lineReceived(self, line):
        count = int(line.strip())
        log.msg('Client requested {} random integers'.format(count))

        producer = Producer(self, count)
        self.transport.registerProducer(producer, True)
        producer.resumeProducing()

    def connectionLost(self, reason=connectionDone):
        print('Connection lost from {}'.format(self.transport.getPeer()))