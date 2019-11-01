import subprocess
from queue import Queue
import time
import numpy as np

from utility.non_blocking_stream_reader import NonBlockingStreamReader as StreamReader
import utility.live_helpers as lh
from tfmodel.predictor import Predictor
from utility.time_quantizer import TimeQuantizer
from visualizer.visualizer import Visualizer

SMILExtract = '../opensmile-2.3.0/inst/bin/SMILExtract'
smile_config = 'smileconfig/live_ComParE_2016_reduced.conf'

config_str_a = 'run2_A_size300_step100_bs16_sl10_nl2_ss80_no10_std_0.9'
config_str_v = 'V_size300_step100_bs16_sl30_nl2_ss80_no10'

model_a = 'golden_models/{}/model.ckpt-{}'.format(config_str_a, 60)
model_v = 'golden_models/{}/model.ckpt-{}'.format(config_str_v, 80)

from zope.interface import implementer
from random import randrange

from twisted.internet import interfaces
from twisted.python import log
from twisted.internet.protocol import Protocol, connectionDone
from twisted.protocols.basic import LineReceiver

import sys
from twisted.python import log
from twisted.internet import reactor

from twisted.internet.protocol import ServerFactory


def load_model():
    return Predictor(model_a, model_v, batch_size=1)


def reload_pa_loopback(latency_ms=580):
    # delay the audio output in order to sync audio and visuals
    subprocess.run(['pacmd', 'unload-module module-loopback'], check=True)
    subprocess.run(['pacmd', 'load-module module-loopback latency_msec={}'.format(latency_ms)], check=True)


def main(predictor=None):
    if predictor is None:
        p = Predictor(model_a, model_v, batch_size=1)
    else:
        p = predictor

    smile_extract = subprocess.Popen([SMILExtract, '-C', smile_config], stdout=subprocess.PIPE)
    # the first two lines are the names of the features
    lld_list = lh.make_feature_list_from_smileout(smile_extract.stdout.readline())
    func_list = lh.make_feature_list_from_smileout(smile_extract.stdout.readline())
    # has not been thrown yet
    assert (len(lld_list) < len(func_list)), 'Funcs initialized before LLDS'

    llds = Queue()
    funcs = Queue()
    arousal = Queue()
    valence = Queue()

    reload_pa_loopback()

    StreamReader(smile_extract.stdout, llds, funcs, len(lld_list))

    visualizer = Visualizer(lld_list, std=0.2)
    visualizer.update_base_color(np.random.rand(), np.random.rand())

    p.start_predicting(funcs, arousal, valence)

    while True:
        if not llds.empty():
            # print('LLds queue size', llds.qsize())
            if llds.qsize() > 5:
                for _ in range(4):
                    llds.get()

            visualizer.update_visuals(llds.get())

        if not arousal.empty() and not valence.empty():
            a = np.float(arousal.get() / 1000)
            v = np.float(valence.get() / 1000)
            # print('Arousal: {}, Valence: {}'.format(a, v))
            visualizer.update_base_color(a, v)
        else:
            time.sleep(0.005)


@implementer(interfaces.IPushProducer)
class Producer(object):
    def __init__(self, proto):
        self._proto = proto
        self._p = Predictor(model_a, model_v, batch_size=1)
        self._paused = False

        self.llds = Queue()
        self.funcs = Queue()
        self.arousal = Queue()
        self.valence = Queue()

    def resumeProducing(self):
        self._paused = False

        self.smile_extract = subprocess.Popen([SMILExtract, '-C', smile_config], stdout=subprocess.PIPE)
        # the first two lines are the names of the features
        lld_list = lh.make_feature_list_from_smileout(self.smile_extract.stdout.readline())
        func_list = lh.make_feature_list_from_smileout(self.smile_extract.stdout.readline())
        # has never been thrown yet
        assert (len(lld_list) < len(func_list)), 'Funcs initialized before LLDS'

        StreamReader(self.smile_extract.stdout, self.llds, self.funcs, len(lld_list))
        visualizer = Visualizer(lld_list, std=0.2, tcp_protocol=self._proto)
        visualizer.update_base_color(np.random.rand(), np.random.rand())

        self._p.start_predicting(self.funcs, self.arousal, self.valence)

        tq = TimeQuantizer()

        while not self._paused:
            if not self.llds.empty():
                # print('LLds queue size', llds.qsize())
                if self.llds.qsize() > 5:
                    for _ in range(4):
                        self.llds.get()

                visualizer.update_visuals(self.llds.get())

            if not self.arousal.empty() and not self.valence.empty():
                a = np.float(self.arousal.get() / 1000)
                v = np.float(self.valence.get() / 1000)
                # print('Arousal: {}, Valence: {}'.format(a, v))
                visualizer.update_base_color(a, v)
            else:
                time.sleep(0.005)

    def pauseProducing(self):
        self._paused = True
        log.msg('Pausing connection from {}'.format(self._proto.transport.getPeer()))

    def stopProducing(self):
        self.smile_extract.kill()
        self._proto.transport.unregisterProducer()
        self._proto.transport.loseConnection()


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


class ProducerFactory(ServerFactory):
    def buildProtocol(self, addr):
        return ServeProducerProtocol()


def run_server():
    log.startLogging(sys.stdout)
    log.msg('Starting Producer Derver')

    reactor.listenTCP(8000, ProducerFactory())
    reactor.run()


if __name__ == '__main__':
    # main(predictor=load_model())
    # tf.app.run(main=main)
    run_server()
