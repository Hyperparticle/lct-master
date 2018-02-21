#!/usr/bin/env bash

# Evaluates the trained model on testing data and outputs accuracy

# Start the test recognition and output to recout.mlf
HVite -T 1 -C configuration/hmm-config -H model/hmm15/macros -H model/hmm15/hmmdefs -S test/test.scp -l '*' -i test/recout.mlf -w test/wdnet.lat -p -20.0 -s 5.0 test/dict.tst train/monophones1

# Evaluate recognition output file recout.mlf by HResults
HResults -t -I test/test_words.mlf train/monophones1 test/recout.mlf