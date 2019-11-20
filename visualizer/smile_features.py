class HLDs:
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

    @staticmethod
    def as_list():
        return [HLDs.entropy, HLDs.centroid, HLDs.flux, HLDs.hnr, HLDs.harmonicity, HLDs.rms,
                HLDs.delta_rms, HLDs.band250_650, HLDs.rolloff]

    @staticmethod
    def get_enables_features_from_config(config, feature_list):
        enabled_features = {}
        for key in config['features']:
            if int(config['features'][key]) == 1 and key in feature_list:
                enabled_features[key] = feature_list.index(key)

        return enabled_features


class EnabledFeatures:
    def __init__(self, config, feature_list):
        self._enabled_features = HLDs.get_enables_features_from_config(config, feature_list)

    def list(self):
        return self._enabled_features

    def get_features(self, feature_maxima, llds):
        flux = llds[self._enabled_features[HLDs.flux]]
        centroid = llds[self._enabled_features[HLDs.centroid]]
        rms = llds[self._enabled_features[HLDs.rms]]
        entropy = llds[self._enabled_features[HLDs.entropy]]
        flux_max = feature_maxima[self._enabled_features[HLDs.flux]]
        energy_max = feature_maxima[self._enabled_features[HLDs.rms]]
        energy_delta = llds[self._enabled_features[HLDs.delta_rms]]
        spect_rolloff = llds[self._enabled_features[HLDs.rolloff]]
        # hnr = llds[self.enabled_features[HLD.smile_hnr]]
        spect_harm = llds[self._enabled_features[HLDs.harmonicity]]

        return flux, centroid, rms, entropy, flux_max, energy_max, energy_delta, spect_rolloff, spect_harm

    def get_mfccs(self, llds):
        mfccs = [llds[self._enabled_features[mf]] for mf in HLDs.mfccs]
        return mfccs

    def get_rastas(self, llds):
        rastas = [llds[self._enabled_features[ra]] for ra in HLDs.rastas]
        return rastas
