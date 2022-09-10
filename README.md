# wav2kana
wav2letter model that outputs the hiragana or katakana transcript of a Japanese language utterance

# Dataset:

A simple dataset to train and test the model is the KORE and Tatoeba datasets. These datasets are usually used to create material to learn Japanese.
However, since Japanese voice recordings with their transcripts are difficult to obtain, they constitute a good start to learn Japanese voice recognition.

Here is a [list of datatests](https://qiita.com/yarimoto/items/98711f23f90ea068730b) you may consider using.



# Environment:

Python 3.7+
PyTorch

sudo apt install ffmpeg

# Prepare the data

`python prepare_data.py -d [DATA_DIR] -r kore_words.csv -a kore-sound-vocab-munged`

`python prepare_data.py -d [DATA_DIR] -r kore_sentences.csv -a kore-sound-sentences-munged`

`python prepare_data.py -d [DATA_DIR] -r tatoeba.csv -a tatoeba_audio`

# Run the training

## Examples

`python train.py -s1 [DATA_DIR1] -r1 .7`

`python train.py -s1 [DATA_DIR1] -r1 1.0 -s2 [DATA_DIR2] -r2 .7 -a 280000 -t 39`

`python train.py -s1 [DATA_DIR1] -r1 1.0 -s2 [DATA_DIR2] -r2 .7 -a 120000 -t 23`

`python train.py -s1 [DATA_DIR1] -r1 1.0 -s2 [DATA_DIR2] -r2 .9 -s3 [DATA_DIR3] -r3 .3 -a 280000 -t 39`





