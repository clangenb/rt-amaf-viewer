import subprocess
import os
import numpy as np

from utility.mkdir_p import mkdir_p
import utility.mirex_data_handlers as mh


if __name__ == '__main__':
    path_to_music = "../mirexdatabase/clips_45seconds/"
    list_csv = os.listdir(path_to_music)
    list_csv.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    smilepath = '../opensmile_new/'
    SMILExtract = 'SMILExtract'
    # smile_config = 'ComParE_2016_reduced_noreg.conf'
    smile_config = 'gemaps/eGeMAPSv01a.conf'

    config_path = 'smileconfig/{}'.format(smile_config)
    smile_save = '../features/{}_fsize300_fstep100/'.format(smile_config.replace('.conf', ''))
    combined_save = '{}reduced/'.format(smile_save)
    mkdir_p(smile_save)
    mkdir_p(combined_save)

    arousal = mh.get_labels_arousal()
    arousal_std = mh.get_std_arousal()
    valence = mh.get_labels_valence()
    valence_std = mh.get_std_valence()

    corrupted_list = []
    i = 0
    for wave_file in list_csv:

        file_path = path_to_music + wave_file
        csv_name = wave_file.replace('.wav', '.csv')
        print('SmileExtract: Processing File', wave_file)

        smile_extract = subprocess.call([SMILExtract,
                                         '-C', config_path,
                                         '-I', file_path,
                                         '-F', smile_save + csv_name])

        feature_set = np.genfromtxt(smile_save + csv_name, delimiter=';', skip_header=1)

        combined = mh.combine_file(feature_set[150::5, 2:],   # 148 at 14,8 seconds
                                   arousal[i, :-1].reshape(-1, 1),
                                   valence[i, :-1].reshape(-1, 1),
                                   arousal_std[i, :-1].reshape(-1, 1),
                                   valence_std[i, :-1].reshape(-1, 1))

        if combined is not None:
            print('Saving File:', csv_name)
            np.savetxt(combined_save + csv_name, combined, delimiter=',')
        else:
            print('Dimensionality Error probably due to corrupted wave in file:', csv_name)
            corrupted_list.append(csv_name)

        i = i+1
    print('Corrupted List: ', corrupted_list)
