# Speech Recognition and Generation (NPFL038)

Author: Dan Kondratyuk
Date: 2018-01-02

The following project uses the HTK toolkit to recognize speech data (i.e., translate speech to text). The provided scripts train a Hidden Markov Model (HMM) on sample test data (audio sentences with their transcriptions) and evaluates this model's ability to transcribe unseen test data.

## Final Results

Upon evaluation, the final output should be as follows:
```
====================== HTK Results Analysis =======================
  Date: Tue Jan  2 17:02:28 2018
  Ref : test/test_words.mlf
  Rec : test/recout.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=40.00 [H=8, S=12, N=20]
WORD: %Corr=92.00, Acc=88.00 [H=161, D=0, S=14, I=7, N=175]
===================================================================
```

We see words were predicted correctly 92% of the time.

## Building and Running

The provided repository contains a pre-trained model that can be evaluated with
```
$ bash ./eval.sh
```

To re-run all training steps, run
```
$ bash ./train.sh
```

To re-run all testing steps, run
```
$ bash ./test.sh
```

## Overview

This section outlines the steps taken in this project.

### Generating Sentences

The command `$ ./scripts/reset.sh` will generate our training and test sentences in the following manner:

1. Clean up all generated files

2. The file `configuration/wdnet.grm` contains the grammar that will generate all sentences. All sentences contain a subject and verb, with optional adjectives and a prepositional phrase. Use `HParse` to create `wdnet.lat` for HTK to read.

3. Generate 40 training sentences and 20 test sentences and output them to `raw-sentences.txt` in the `train` and `test` folders respectively.

### Recording Sentences

1. Record all train/test sentences with a suitable microphone and put them in the `raw-audio` directories.

2. Convert the `.m4a` files to `.wav` using `avconv` or `ffmpeg` using the script `./scripts/convert.sh` (48kHz 16bit). This will also put the `.wav` files in the `converted-audio` directories. They will be named `S001.wav`, `S002.wav`, ...

### Training

The command `$ bash ./train.sh` will run all necessary training steps. It first generates Generates word lists, dictionaries, and feature vector files from the training data. Then it trains an HMM in a sequence: flat start monophones, silence model with short pauses, forced alignment, and triphones. The final HMM model for testing will be outputted to `./model/hmm15`.

The training script takes the following steps (see the script itself for more detail). All generated training files are in the `train` directory.

1. Generate transcription file `words.mlf` from generated sentences for HTK to consume

2. Generate conversion list `codestr.scp` and MFC list `train.scp`

3. Convert WAVs to MFC feature vector files using `./configuration/mfc-config`

4. Create a dictionary that translates from words to phonemes. This project uses an English lexicon provided by Voxforge (see script). A list of words `wlist` is generated and fed through the lexicon to generate a `dict` and list of monophones `monophones0` (without a short pause sp), `monophones1` (with short pause sp).

5. Use script `configuration/mkphones0.led` to create a phonetical transcription of all training sentences from `words.mlf` and output to `phones0.mlf`

6. Define a prototype model in `configuration/proto` and initialize it in `hmm0` by computing the mean and variance of the training data.

7. Generate states from the prototype model for all phonemes in `monophones0` and output to `hmm0/hmmdefs`

8. Train first three training cycles `hmm1`-`hmm3` using Baum-Welch training on flat start monophones

9. Train `hmm4`-`hmm5` with a silence model modification. First copy `hmm3` to `hmm4`, and in `hmm4/hmmedefs` manually create one-state sp model with parameters of middle state of sil model. Then tie the sp state to the sil center state and output to `hmm5`.

10. Generate a new phonetic transcription from `words.mlf` but this time with short pauses between words and output to `phones1.mlf`

11. Run two more training cycles in `hmm6`-`hmm7` which now contain the short pauses

12. Extend dictionary `dict` to include a word for silence and output to `dict2`

13. Run forced-alignment on `hmm7` which will allow the training data to properly align states with their phonemic outputs. This will generate `aligned.mlf`.

14. Run two more training cycles `hmm8`-`hmm9` on the new aligned data.

15. Use script `configuration/mktri.led` to create a triphone transcription on the aligned sentence data and output to `wintri.mlf`, `triphones1`

16. Create `mktri.hed`. When reestimating these new tied parameters the data from each of the original untied parameters is pooled so that a better estimate can be obtained. 

17. Run three more training cycles `hmm10`-`hmm12` to bind monophone to triphone model

18. Make a tied-state triphone dictionary `dict-tri`. Append the contents of monophones0 to the beginning of to the `fulllist0` file, and then to to remove any duplicate entries, and put the result in `fulllist`

19. Copy `configuration/tree1.hed` to `tree.hed`, append state clusters to `tree.hed`

20. Run three more training cycles `hmm13`-`hmm15` on the triphone model and append "uw" state to end of `hmm15/hmmdefs`

### Testing and Evaluation

The command `$ bash ./test.sh` will run all necessary testing steps. Like `train.sh`, it will generate word lists, dictionaries, and feature vector files, but this time on the test data. Finally, the model will predict the output sentences from input audio and evaluate for accuracy.

The testing script takes the following steps (see the script itself for more detail). All generated training files are in the `test` directory.

1. Generate transcription file `test_words.mlf`

2. Generate conversion list `test_codestr.scp` and MFC list `test.scp`

3. Convert WAVs to MFC feature vector files using `./configuration/mfc-config`

4. Create a dictionary that translates from words to phonemes and output `dict.tst`

5. Start the test recognition using model `hmm9` and test data audio and output results to `recout.mlf`

6. Evaluate predicted output to actual output and output final accuracy
