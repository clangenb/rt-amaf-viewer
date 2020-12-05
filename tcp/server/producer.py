from zope.interface import implementer

import subprocess
from queue import Queue
import time
import numpy as np

from utility.non_blocking_stream_reader import NonBlockingStreamReader as StreamReader
import utility.live_helpers as lh
from tf.predictor import Predictor
from utility.time_quantizer import TimeQuantizer
from visualizer.matrix.leds.tcp_strips import LazyTcpStrip, TcpStrip
from visualizer.visualizer import Visualizer

from twisted.python import log
from twisted.internet import interfaces

from params import *

@implementer(interfaces.IPushProducer)
class Producer(object):
    def __init__(self, proto):
        self._proto = proto
        self._p = Predictor(model_a, model_v, batch_size=1)
        self._paused = False
        self.smile_extract = None
        self.llds = Queue()
        self.funcs = Queue()
        self.arousal = Queue()
        self.valence = Queue()
        self.visualizer = None

    def resumeProducing(self):
        self._paused = False

        if self.smile_extract is None:
            self.smile_extract = subprocess.Popen([SMILExtract, '-C', smile_config], stdout=subprocess.PIPE)
            # the first two lines are the names of the features
            lld_list = lh.make_feature_list_from_smileout(self.smile_extract.stdout.readline())
            func_list = lh.make_feature_list_from_smileout(self.smile_extract.stdout.readline())
            # has never been thrown yet
            assert (len(lld_list) < len(func_list)), 'Funcs initialized before LLDS'

            StreamReader(self.smile_extract.stdout, self.llds, self.funcs, len(lld_list))
            self.visualizer = Visualizer(lld_list, std=0.2, led_strip=TcpStrip(self._proto))
            self.visualizer.update_base_color(np.random.rand(), np.random.rand())

            self._p.start_predicting(self.funcs, self.arousal, self.valence)

        timer = TimeQuantizer()
        while not self._paused:
            if not self.llds.empty():
                # print('LLds queue size', llds.qsize())
                self.reduce_queue_size(self.llds)

                self.visualizer.update_visuals(self.llds.get())

            if not self.arousal.empty() and not self.valence.empty():
                self.reduce_queue_size(self.arousal)
                self.reduce_queue_size(self.valence)

                if timer.measure_total() > 0.5:
                    timer.reset()
                    update_base_color_counter = 0
                    a = np.float(self.arousal.get() / 1000)
                    v = np.float(self.valence.get() / 1000)
                    # print('Arousal: {}, Valence: {}'.format(a, v))
                    self.visualizer.update_base_color(a, v)
            else:
                time.sleep(0.005)

    @staticmethod
    def reduce_queue_size(queue):
        if queue.qsize() > 5:
            for _ in range(4):
                queue.get()

    def pauseProducing(self):
        self._paused = True

    def stopProducing(self):
        if self.smile_extract is not None:
            self.smile_extract.kill()

        log.msg('Stop Producing')
        self._proto.transport.unregisterProducer()
        self._proto.transport.loseConnection()
