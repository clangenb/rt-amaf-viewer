#############################################################
# Predictor for live performance, spawns a thread that awaits
# the features in the funcs queue
##############################################################


from threading import Thread
import time

from tfmodel.testmodel import TestModel


# default values for command line use
config_str_a = 'run2_A_size300_step100_bs16_sl10_nl2_ss80_no10_std_0.9'
config_str_v = 'V_size300_step100_bs16_sl30_nl2_ss80_no10'

model_a = 'golden_models/{}/model.ckpt-{}'.format(config_str_a, 60)
model_v = 'golden_models/{}/model.ckpt-{}'.format(config_str_v, 80)


class Predictor:
    def __init__(self, model_arousal=model_a, model_valence=model_v, batch_size=1):
        self._arousal = TestModel(model_arousal, batch_size, namescope='model_A')
        self._valence = TestModel(model_valence, batch_size, namescope='model_V')

    def start_predicting(self, funcs_queue, arousal_queue, valence_queue):
        t = Thread(target=self._predict, args=(funcs_queue, arousal_queue, valence_queue))
        t.daemon = True
        t.start()

    def _predict(self, funcs, arousal_queue, valence_queue):
        while True:
            if not funcs.empty():
                x = funcs.get()
                # print('Funcs size: ', funcs.qsize())

                arousal = self._arousal.predict(x)
                valence = self._valence.predict(x)

                arousal_queue.put(arousal)
                valence_queue.put(valence)
            else:
                time.sleep(0.300)
