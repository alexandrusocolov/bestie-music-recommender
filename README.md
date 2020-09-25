# Bestie
 
**Bestie** is a command-line Python utility for recommending songs similar to an input. The tool is powered by a pre-trained neural network [MusicNN (Pons, 2019)](https://github.com/jordipons/musicnn).

## Usage

```
python3 recommend.py path_to_file.mp3 
```

Where `path_to_file.mp3` is a path to the audio you want to get similar songs for. After running this command, **Bestie** will create a folder called `your_recommendations` with the recommended songs. The rank from most recommended (1) till least (3 by default, changable in `recommend.py`) is at the end of the file name, e.g. `troye1_n2.mp3` means song called `troye1.mp3` is second-most suggested.

## How it works
**Bestie** uses [MusicNN](https://github.com/jordipons/musicnn) taggram features, i.e. a vector of 50 probabilities corresponding to how expressed a certain tag like 'techno' or 'no voice' is. The full list of tags can be found [here](https://github.com/jordipons/musicnn/blob/master/FAQs.md). These features are pre-computed using `create_dataset.py` for the database of songs out of which **Bestie** can recommend (currently 15 audios) from the `music` folder. The pre-computed MusicNN features are stored in `data/train_set.csv`.

All in all, for a given path to an audio provided in the command line:
- Extract MusicNN features
- Compute cosine similarity in MusicNN features with the recommendable songs in the `music` folder
- Copy n (by default 3) most similar audios into `your_recommendation` folder 


## FAQs
*What to do if I want to add more audios?*
Add them to `music` folder and re-run `create_dataset.py`.

*How do I recommend more audios than just 3?*
Change the `top_n` parameter in `recommend.py`.

*Can I get a recommendation for a song that is not in `mp3` format?*
Unfortunately, the script has only been tested on `mp3` files so I cannot promise a stable performance for anything other than `mp3`.
