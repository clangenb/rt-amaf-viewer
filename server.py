import subprocess

from tcp.tcp_server.server_factory import ProducerFactory

import sys
from twisted.python import log
from twisted.internet import reactor


def reload_pa_loopback(latency_ms=580):
    # delay the audio output in order to sync audio and visuals
    subprocess.run(['pacmd', 'unload-module module-loopback'], check=True)
    subprocess.run(['pacmd', 'load-module module-loopback latency_msec={}'.format(latency_ms)], check=True)


def run_server():
    log.startLogging(sys.stdout)
    log.msg('Starting Producer Server')

    reactor.listenTCP(8000, ProducerFactory())
    reactor.run()


if __name__ == '__main__':
    run_server()
