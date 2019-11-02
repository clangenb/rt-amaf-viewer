from __future__ import print_function

import sys
from twisted.python import log
from twisted.internet import reactor
from tcp.tcp_client.client_factory import EchoClientFactory, RandomReceiverFactory, LedReceiverFactory


# this connects the protocol to a server running on port 8000
def main():
    log.startLogging(sys.stdout)
    log.msg('Starting TCP client')

    # reactor.connectTCP("localhost", 8000, EchoClientFactory())
    reactor.connectTCP("192.168.1.23", 8000, LedReceiverFactory())
    reactor.run()


# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()
