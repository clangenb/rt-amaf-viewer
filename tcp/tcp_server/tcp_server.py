# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.

import sys
from twisted.python import log
from twisted.internet import reactor

from tcp.tcp_server.server_factory import EchoServerFactory, RandomServerFactory

def main():
    log.startLogging(sys.stdout)
    log.msg('Starting TCP Server')

    """This runs the protocol on port 8000"""
    # reactor.listenTCP(8000, EchoServerFactory())
    reactor.listenTCP(8000, RandomServerFactory())
    reactor.run()


if __name__ == '__main__':
    main()