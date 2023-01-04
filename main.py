import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob
from os.path import basename

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audio_files = glob('./audio_files/*.wav')

# Play audio file
ipd.Audio(audio_files[0])


audio_file = audio_files[1]


def show_audio_data(audio_f):
    y, sr = librosa.load(audio_f)
    pd.Series(y).plot(figsize=(10, 5), lw=1, title='Raw Audio Example', color=color_pal[0])
    plt.show()


def show_spectrogram(audio_f):
    y, sr = librosa.load(audio_f)
    d = librosa.stft(y)
    s_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)

    # Plot the transformed audio data
    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(s_db, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(f'Spectrogram of "{basename(audio_file)}"', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()


show_spectrogram(audio_file)
