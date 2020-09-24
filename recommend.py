import math, os, sys
import numpy as np
import pandas as pd
import warnings
from utils import cosine_similarity
from mutagen.mp3 import MP3
from musicnn.extractor import extractor

# ignore all warnings
warnings.filterwarnings("ignore")

# check if an input was provided
if len(sys.argv) != 2:
    print("Wrong usage. Correct example: `python3 recommend.py path_to_file.mp3`")
    sys.exit(1)

# get the argument/path provided and check if it's .mp3
input_path = sys.argv[1]
if '.mp3' not in input_path:
    print("Wrong path format. The script has only been tested on .mp3 files")
    sys.exit(2)

# get the length of the audio
input_length = math.floor(MP3(input_path).info.length)

# extract features using MusicNN
input_embed, _, _ = extractor(input_path,
                                    model='MTT_musicnn',
                                    input_length=input_length,
                                    extract_features=True)

# load in the training set and select the embedding part
data = pd.read_csv('data/train_set.csv')
embeddings = data.iloc[:, 1:].values


# compute the similarities between input_embed and embeddings
cos_sim = []
for i in range(embeddings.shape[0]):
    cos_sim.append(cosine_similarity(input_embed.tolist()[0], embeddings[i, :].tolist()))

# sort cosine similarities, get top_n of them and their audio paths
top_n = 3
idx = np.array(cos_sim).argsort()[::-1][:top_n]
chosen_audio_paths = [data.audio_path.values[id] for id in idx]

# create a folder with the recommended audios
os.system("rm -r your_recommendation")
os.system("mkdir your_recommendation")
for i, audio_path in enumerate(chosen_audio_paths):
    os.system('cp ' + audio_path + ' ' + audio_path.replace('music', 'your_recommendation').replace('.', '_n'+ str(i+1) + '.'))

print("Your recos are ready!")
sys.exit(0)
