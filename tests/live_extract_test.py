import subprocess
import tensorflow as tf
from queue import Queue
import time
import numpy as np

from utility.non_blocking_stream_reader import NonBlockingStreamReader as StreamReader
from utility.time_quantizer import TimeQuantizer
import utility.live_helpers as lh
from visualizer.visualizer import Visualizer


def main(_):
    smile_extract = subprocess.Popen([SMILExtract, '-C', smile_config], bufsize=1, stdout=subprocess.PIPE)

    # the first to lines are the names of the features
    lld_list = lh.make_feature_list_from_smileout(smile_extract.stdout.readline())
    print(lld_list)
    func_list = lh.make_feature_list_from_smileout(smile_extract.stdout.readline())
    assert (len(lld_list) < len(func_list)), 'Funcs initialized before LLDS'

    # delay the audio output in order to sync audio and visuals
    subprocess.run(['pacmd', 'unload-module module-loopback'], check=True)
    subprocess.run(['pacmd', 'load-module module-loopback latency_msec=615'], check=True)

    llds = Queue()
    funcs = Queue()

    StreamReader(smile_extract.stdout, llds, funcs, len(lld_list))

    timer = TimeQuantizer()

    visualizer = Visualizer(lld_list, std=0.1)
    visualizer.update_base_color(np.random.rand(), np.random.rand())

    while True:
        timer.reset()
        if not llds.empty():
            visualizer.update_visuals(llds.get())
            print('Total Loop Time: ', timer.measure_total(), 'Queue Size: ', llds.qsize())
            if llds.qsize() > 10:
                for _ in range(5):
                    llds.get()

        elif not funcs.empty():
            funcs.get()
            visualizer.update_base_color(np.random.rand() - 0.5, np.random.rand() - 0.5)
            print('Updating base color time: ', timer.measure_step())
        else:
            print('Sleeping')
            time.sleep(0.003)


if __name__ == '__main__':
    SMILExtract = '../opensmile-2.3.0/inst/bin/SMILExtract'
    smile_config = 'smileconfig/live_ComParE_2016_reduced.conf'
    # smile_config = 'smileconfig/gemaps/live_eGeMAPSv01a.conf'

    config_str_a = 'run2_A_size300_step100_bs16_sl10_nl2_ss80_no10_std_0.9'
    config_str_v = 'V_size300_step100_bs16_sl30_nl2_ss80_no10'

    model_a = 'golden_models/{}/model.ckpt-{}'.format(config_str_a, 60)
    model_v = 'golden_models/{}/model.ckpt-{}'.format(config_str_v, 80)

    tf.app.run(main=main)
