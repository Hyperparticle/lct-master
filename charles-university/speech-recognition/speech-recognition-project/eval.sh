#!/usr/bin/env bash

# Evaluates the trained model on testing data and outputs accuracy

# Start the test recognition and output to recout.mlf
HVite -T 1 -C configuration/hmm-config -H model/hmm9/macros -H model/hmm9/hmmdefs -S test/test.scp -l '*' -i test/recout.mlf -w test/wdnet.lat -p 0.0 -s 5.0 configuration/voxforge_lexicon_dict train/monophones1

# Evaluate recognition output file recout.mlf by HResults
HResults -I test/test_words.mlf train/monophones1 test/recout.mlf