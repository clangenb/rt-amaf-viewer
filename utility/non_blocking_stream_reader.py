from threading import Thread
from utility.live_helpers import make_feature_vector_from_bytecode_string
import time

class NonBlockingStreamReader:

    def __init__(self, stream, lld_queue, func_queue, num_llds):
        self._s = stream
        self._lld_queue = lld_queue
        self._func_queue = func_queue
        self._num_llds = num_llds

        def _populate_queue(stream, llds, funcs):

            while True:
                line = stream.readline()
                if line:
                    features = make_feature_vector_from_bytecode_string(line)
                    if features.shape[-1] <= self._num_llds:
                        llds.put(features)
                    else:
                        funcs.put(features)
                else:
                    # print('No openSMILE output yet going to sleep')
                    time.sleep(0.004)

        self._t = Thread(target=_populate_queue, args=(self._s, self._lld_queue, self._func_queue))
        self._t.daemon = True
        self._t.start()


# class UnexpectedEndOfStream(Exception):
#     print('No OpenSMILE output (yet)!')
