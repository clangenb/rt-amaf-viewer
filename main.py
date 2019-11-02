import subprocess
from queue import Queue
import time
import numpy as np

from tcp.tcp_server.server_factory import ProducerFactory
from utility.non_blocking_stream_reader import NonBlockingStreamReader as StreamReader
import utility.live_helpers as lh
from tfmodel.predictor import Predictor
from visualizer.visualizer import Visualizer

from params import *

import sys
from twisted.python import log
from twisted.internet import reactor

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


def run_server():
    log.startLogging(sys.stdout)
    log.msg('Starting Producer Derver')

    reactor.listenTCP(8000, ProducerFactory())
    reactor.run()


if __name__ == '__main__':
    # main(predictor=load_model())
    # tf.app.run(main=main)
    run_server()
