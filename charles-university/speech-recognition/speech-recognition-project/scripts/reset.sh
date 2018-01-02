#!/usr/bin/env bash

# Resets the entire project, removing all files and generating new sentences to record from the grammar

# Clean up all files
rm -rf model/hmm*
cd train
rm -f converted-audio/* raw-audio/*
rm -f sentences.txt words.mlf codestr.scp train.scp vocab.txt wlist dlog dict monophones0 monophones1 phones0.mlf phones1.mlf dict2 aligned.mlf
cd ../test
rm -f converted-audio/* raw-audio/*
rm -f sentences.txt wdnet.lat test_words.mlf wlist_test dict.tst dlog test_codestr.scp test.scp recout.mlf vocab.txt
cd ..

# Generate new train and test sentences (40/20)
HParse configuration/wdnet.grm test/wdnet.lat
HSGen -n 40 test/wdnet.lat configuration/voxforge_lexicon_dict | nl -nln -d'\t' > train/raw-audio/raw-sentences.txt
HSGen -n 20 test/wdnet.lat configuration/voxforge_lexicon_dict | nl -nln -d'\t' > test/raw-audio/raw-sentences.txt
