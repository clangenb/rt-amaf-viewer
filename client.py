from __future__ import print_function

import sys
from twisted.python import log
from twisted.internet import reactor
from tcp.tcp_client.client_factory import LedReceiverFactory


def main():
    log.startLogging(sys.stdout)
    log.msg('Starting TCP client')

    reactor.connectTCP("192.168.1.23", 8000, LedReceiverFactory())
    reactor.run()


if __name__ == '__main__':
    main()
