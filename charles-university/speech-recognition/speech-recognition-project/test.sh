#!/usr/bin/env bash

cd test

# Clean up extraneous files
rm -f converted-audio/*.mfc
rm -f sentences.txt wdnet.lat test_words.mlf wlist_test dict.tst dlog test_codestr.scp test.scp recout.mlf vocab.txt

# 1. Test of models
# Write your test grammar to file wdnet.grm and compile it into HTK format: 
# Create phonetical dictionary dict.tst from your grammar words and generate test sentences:
HParse ../configuration/wdnet.grm wdnet.lat
cat raw-audio/raw-sentences.txt | sed -E 's/^.*\t//g' > sentences.txt

cat raw-audio/raw-sentences.txt | sed -E 's/[[:punct:]]/ /g' | tr '[:lower:]' '[:upper:]' | sed 's/^ //g' > vocab.txt
julia ../scripts/prompts2wlist.jl vocab.txt wlist_test
HDMan -A -D -T 1 -m -w wlist_test -i -l dlog -g ../configuration/global.ded dict.tst ../configuration/voxforge_lexicon_dict

# 3. Record 20 sentences to wav files in the same way as training sentences and create analogically with help of previous scripts following files: 
# test_codestr.scp, test.scp, test_words.mlf
# Generate transcription file from sentences with following script (toMLF.prl, words.mlf)
cat sentences.txt | sed -E 's/[[:punct:]]/ /g' | grep . | perl ../scripts/toMLF.prl > test_words.mlf

# Generate conversion and MFC list (codestr.scp, train.scp)
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\(.*\).wav/\1.wav \1.mfc/" > test_codestr.scp
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\.wav/\.mfc/" > test.scp

# Convert WAVs to MFC feature vector files
HCopy -T 1 -C ../configuration/mfc-config -S test_codestr.scp

cd ..

# 4. Start your test recognition:
bash ./eval.sh
