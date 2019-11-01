import time
import numpy as np

from visualizer.test_visualizer import Visualizer

smile_entropy = 'pcm_fftMag_spectralEntropy_sma'
smile_centroid = 'pcm_fftMag_spectralCentroid_sma'
smile_flux = 'pcm_fftMag_spectralFlux_sma'
smile_hnr = 'logHNR_sma'
smile_harmonicity = 'pcm_fftMag_spectralHarmonicity_sma'
smile_rms = 'pcm_RMSenergy_sma'
smile_delta_rms = 'pcm_RMSenergy_sma_de'
smile_band250_650 = 'pcm_fftMag_fband250-650_sma'
smile_rolloff = 'pcm_fftMag_spectralRollOff75.0_sma'


def main():
    emotion_update = 1
    lld_list = [smile_band250_650, smile_centroid, smile_delta_rms, smile_rms, smile_entropy, smile_harmonicity, smile_flux,
                smile_hnr, smile_rolloff]
    visualizer = Visualizer(lld_list, std=0.2)

    visualizer.update_base_color(np.random.rand(), np.random.rand())
    i = 0
    while i < 100:
        llds = np.random.rand(len(lld_list))
        visualizer.update_visuals(llds)

        if emotion_update % 100 == 0:
                visualizer.update_base_color(np.random.rand(), np.random.rand())
                emotion_update = 0
        emotion_update = emotion_update + 1
        i = i +1
        time.sleep(0.5)


if __name__ == '__main__':
    main()
