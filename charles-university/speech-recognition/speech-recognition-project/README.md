# Speech Recognition and Generation (NPFL038)

## Building and Running

### Steps

1. Create grammar, generate 40 training sentences, 20 test sentences
Run `$ bash ./scripts/reset.sh` to remove all files and generate.

2. Record 40  training sentences (name them for example S001.wav, .., S040.wav) with audacity (48kHz 16bit, WAV) or with following script (recordIt.prl)

3. Run `$ bash ./scripts/convert.sh` to convert .m4a files to .wav files

#### Training

4. Run training script `$ bash ./train.sh`

The training script takes the following steps

#### Testing

#### Evaluation

# For some languages you can define your own transcription rules from orthographical to phonetical representation.
# Or you can download some online dictionaries for your language or use the online MaryTTs phonetical output (if your language is included). 
# Then extract word list(dict.tmp) and create dictionary file dict:
