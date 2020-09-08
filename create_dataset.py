import math
import os
import numpy as np
import pandas as pd
from musicnn.extractor import extractor
from musicnn.tagger import top_tags
from mutagen.mp3 import MP3

audio_names = os.listdir('music/')
audio_paths = ['music/' + i for i in audio_names]

# creating a data set with MusicNN features
features = []
for audio_path in audio_paths:
    print('Extracting features for ' + audio_path)

    # check the length of the audio
    audio_length = math.floor(MP3(audio_path).info.length)

    # extract features using MusicNN
    taggram, tags, _ = extractor(audio_path,
                              model='MTT_musicnn',
                              input_length=audio_length,
                              extract_features=True)
    print(taggram.shape)
    features.append(taggram)

features = np.vstack(features)

# create and save a table
df = pd.DataFrame(features, columns=tags)
df['audio_path'] = audio_paths
df = df[['audio_path'] + tags]   # rearrange columns
df.to_csv('data/train_set.csv', index=False)




