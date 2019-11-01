from zope.interface import implementer
from random import randrange

from twisted.python import log
from twisted.internet import interfaces

@implementer(interfaces.IPushProducer)
class Producer(object):
    def __init__(self, proto, count):
        self._proto = proto
        self._goal = count
        self._produced = 0
        self._paused = False

    def pauseProducing(self):
        self._paused = True
        log.msg('Pausing connection from {}'.format(self._proto.transport.getPeer()))

    def resumeProducing(self):
        self._paused = False

        while not self._paused and self._produced < self._goal:
            next_int = randrange(0, 19999)
            line = "{}".format(next_int)
            self._proto.sendLine(line.encode("ascii"))
            self._produced += 1

        if self._produced == self._goal:
            self._proto.transport.unregisterProducer()
            self._proto.transport.loseConnection()

    def stopProducing(self):
        self._produced = self._goal
