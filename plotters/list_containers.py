import numpy as np
from scipy import interpolate


class ErrorListContainer:
    """
    Class that stores the errors and has some basic plotting functionality
    """
    def __init__(self):
        self.l1_a = []
        self.rms_a = []
        self.l1_v = []
        self.rms_v = []

    def extend(self, l1a, rmsa, l1v, rmsv):
        self.l1_a.extend(l1a)
        self.rms_a.append(rmsa)
        self.l1_v.extend(l1v)
        self.rms_v.append(rmsv)

    def plot(self, axes, xmax):
        x = np.linspace(0, xmax, len(self.l1_a))
        axes[0].plot(x, self.l1_a)
        x = np.linspace(0, xmax, len(self.rms_a))
        axes[0].plot(x, self.rms_a)
        axes[0].legend(['L1-Error', 'RMSE'])

        x = np.linspace(0, xmax, len(self.l1_v))
        axes[1].plot(x, self.l1_v)
        x = np.linspace(0, xmax, len(self.rms_v))
        axes[1].plot(x, self.rms_v)
        axes[1].legend(['L1-Error', 'RMSE'])


class PredictionListContainer:
    """
    Class that stores the predictions and labels and has some basic
    functionality for simpler animation
    """
    def __init__(self):
        self.pa = []
        self.ya = []
        self.pv = []
        self.yv = []

    def extend(self, pa, ya, pv, yv):
        if hasattr(ya, '__iter__'):
            self.ya.extend(ya)
            self.yv.extend(yv)
            self.pa.extend(pa)
            self.pv.extend(pv)
        else:
            # in rt_prediction_animation.py we only append a float thus extend raises an error
            self.pa.append(pa[0])
            self.pv.append(pv[0])
            self.ya.append(ya)
            self.yv.append(yv)

    def interpolate(self, interp):
        """ Make smooth animations as predictions are only generated at 500 ms interval"""
        x = np.linspace(0, 29, len(self.ya))
        f_ya = interpolate.interp1d(x, self.ya)
        f_yv = interpolate.interp1d(x, self.yv)
        f_pa = interpolate.interp1d(x, np.reshape(self.pa, [-1]))
        f_pv = interpolate.interp1d(x, np.reshape(self.pv, [-1]))

        x_interp = np.linspace(0, 29, len(self.ya)*interp)
        self.ya = list(f_ya(x_interp))
        self.yv = list(f_yv(x_interp))
        self.pa = list(f_pa(x_interp))
        self.pv = list(f_pv(x_interp))

    def frame_generator(self, frame_size, frame_step):
        """ Get frames for the animations"""
        num_frames = frame_size
        while self.ya:
            ya = self.ya[:num_frames]
            del self.ya[:frame_step]
            yv = self.yv[:num_frames]
            del self.yv[:frame_step]
            pa = self.pa[:num_frames]
            del self.pa[:frame_step]
            pv = self.pv[:num_frames]
            del self.pv[:frame_step]

            yield ya, yv, pa, pv

    def clear_lists(self):
        self.pa = []
        self.ya = []
        self.pv = []
        self.yv = []
