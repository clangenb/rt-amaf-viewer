import subprocess
import tensorflow as tf
from queue import Queue
import time
import numpy as np

from utility.non_blocking_stream_reader import NonBlockingStreamReader as StreamReader
import utility.live_helpers as lh
from tf.predictor import Predictor
from visualizer.visualizer import Visualizer
from visualizer.patterns.mood_slide_show import MoodSlider

SMILExtract = '../opensmile-2.3.0/inst/bin/SMILExtract'
smile_config = 'smileconfig/live_ComParE_2016_reduced.conf'

config_str_a = 'run2_A_size300_step100_bs16_sl10_nl2_ss80_no10_std_0.9'
config_str_v = 'V_size300_step100_bs16_sl30_nl2_ss80_no10'

model_a = 'golden_models/{}/model.ckpt-{}'.format(config_str_a, 60)
model_v = 'golden_models/{}/model.ckpt-{}'.format(config_str_v, 80)


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
    # the first to lines are the names of the features
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

    ms = MoodSlider()

    p.start_predicting(funcs, arousal, valence)

    while True:
        if not llds.empty():
            # print('LLds queue size', llds.qsize())
            if llds.qsize() > 10:
                for _ in range(9):
                    llds.get()

                ms.slide()

        if not arousal.empty() and not valence.empty():
            a = np.float(arousal.get() / 1000)
            v = np.float(valence.get() / 1000)
            print('Arousal: {}, Valence: {}'.format(a, v))
            ms.slide(a,v)
        else:
            time.sleep(0.005)


if __name__ == '__main__':
    # main(predictor=load_model())
    tf.app.run(main=main)
