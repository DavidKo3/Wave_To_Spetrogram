# Procsess the data and create spectograms
labels = ['air_conditioner','car_horn','children_playing',
           'dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import numpy as np
from pathlib import Path
import os
import cv2
from tqdm import tqdm
# from scipy.misc import imsave

def create_fold_spectrograms(fold):
    spectrogram_path = Path('/data2/davidk/UrbanSound8K/spectrogram')
    audio_path = Path('/data2/davidk/UrbanSound8K/audio')
    print(f'Processing fold {fold}')
    if not os.path.exists(spectrogram_path):
        os.mkdir(spectrogram_path)
    os.mkdir(spectrogram_path/fold)
    for audio_file in tqdm(list(Path(audio_path/f'fold{fold}').glob('*.wav'))):
        samples, sample_rate = librosa.load(audio_file)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename  = spectrogram_path/fold/Path(audio_file).name.replace('.wav','.png')
        S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')


for i in range(1, 11):
    create_fold_spectrograms(str(i))
