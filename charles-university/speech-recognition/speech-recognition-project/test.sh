#!/usr/bin/env bash

# Run all test steps
# Generates word lists, dictionaries, MFC files from test data
# Evaluates the trained model on test data and outputs accuracy

cd test

# Clean up extraneous files
rm -f converted-audio/*.mfc
rm -f sentences.txt wdnet.lat test_words.mlf wlist_test dict.tst dlog test_codestr.scp test.scp recout.mlf vocab.txt

# Compile test grammar (wdnet.grm) into HTK format (wdnet.lat)
HParse ../configuration/wdnet.grm wdnet.lat

# Create phonetical dictionary dict.tst from the grammar words similar to training
cat raw-audio/raw-sentences.txt | sed -E 's/^.*\t//g' > sentences.txt
cat raw-audio/raw-sentences.txt | sed -E 's/[[:punct:]]/ /g' | tr '[:lower:]' '[:upper:]' | sed 's/^ //g' > vocab.txt
julia ../scripts/prompts2wlist.jl vocab.txt wlist_test
HDMan -A -D -T 1 -m -w wlist_test -i -l dlog -g ../configuration/global.ded dict.tst ../configuration/voxforge_lexicon_dict

# Similar to training, generate transcription, conversion, and MFC list files (test_words.mlf, test_codestr.scp, test.scp)
cat sentences.txt | sed -E 's/[[:punct:]]/ /g' | grep . | perl ../scripts/toMLF.prl > test_words.mlf
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\(.*\).wav/\1.wav \1.mfc/" > test_codestr.scp
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\.wav/\.mfc/" > test.scp

# Convert WAVs to MFC feature vector files
HCopy -T 1 -C ../configuration/mfc-config -S test_codestr.scp

cd ..

# Start test recognition and output accuracy
bash ./eval.sh
