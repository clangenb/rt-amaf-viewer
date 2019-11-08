class HLD:
    entropy = 'pcm_fftMag_spectralEntropy_sma'
    centroid = 'pcm_fftMag_spectralCentroid_sma'
    flux = 'pcm_fftMag_spectralFlux_sma'
    hnr = 'logHNR_sma'
    harmonicity = 'pcm_fftMag_spectralHarmonicity_sma'
    rms = 'pcm_RMSenergy_sma'
    delta_rms = 'pcm_RMSenergy_sma_de'
    band250_650 = 'pcm_fftMag_fband250-650_sma'
    rolloff = 'pcm_fftMag_spectralRollOff75.0_sma'
    mfccs = ['mfcc_sma[{}]'.format(i) for i in range(1, 14)]
    rastas = ['audSpec_Rfilt_sma[{}]'.format(i) for i in range(26)]

