# wav2kana
wav2letter model that outputs the hiragana or katakana transcript of a Japanese language utterance

# Dataset:

A simple dataset to train and test the model is the KORE and Tatoeba datasets. These datasets are usually used to create material to learn Japanese.
However, since Japanese voice recordings with their transcripts are difficult to obtain, they constitute a good start to learn Japanese voice recognition.

The transcripts along with the corresponding audio file names are contained in .csv files that you can find in [this repository](https://drive.google.com/drive/folders/1VFKsG6C2VXHePO9_3X2tN1CHhd6b482C).

The corresponding audio files for the KORE dataset can be downloaded from:

[Single words](http://www.mediafire.com/file/oyddnozmbd2/kore-sound-vocab-munged.zip/file)

[Sentences](http://www.mediafire.com/file/1itzmjondnz/kore-sound-sentences-munged.zip/file)

And in this repository for the [Tatoeba dataset](https://drive.google.com/drive/folders/1VFKsG6C2VXHePO9_3X2tN1CHhd6b482C).

# Environment:

Python 3.7+
PyTorch

sudo apt install ffmpeg

# Prepare the data

`python prepare_data.py -d [DATA_DIR] -r kore_sentences_katakana.csv -a kore-sound-vocab-munged`

`python prepare_data.py -d [DATA_DIR] -r kore_sentences_katakana.csv -a kore-sound-sentences-munged`

`python prepare_data.py -d [DATA_DIR] -r tatoeba_katakana.csv -a tatoeba_audio`




