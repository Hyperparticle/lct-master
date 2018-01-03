#!/usr/bin/env bash

# Run all training steps
# Generates word lists, phoneme translation dictionaries, MFC feature vector files from training data
# Trains an HMM in a sequence: flat start monophones, silence model with short pauses, forced alignment, and triphones
# Outputs final HMM to ./model/hmm15

cd train

# Clean up extraneous files
rm -rf model/hmm*
rm -f converted-audio/*.mfc
rm -f sentences.txt words.mlf codestr.scp train.scp vocab.txt wlist dlog dict monophones0 monophones1 phones0.mlf phones1.mlf dict2 aligned.mlf triphones1 mktri.hed fulllist0 flog dict-tri fulllist tree.hed stats tiedlist wintri.mlf trees

# Generate transcription file (words.mlf) from sentences with toMLF.prl
cat raw-audio/raw-sentences.txt | sed -E 's/^.*\t//g' > sentences.txt
cat sentences.txt | sed -E 's/[[:punct:]]/ /g' | grep . | perl ../scripts/toMLF.prl > words.mlf

# Generate conversion and MFC list (codestr.scp, train.scp)
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\(.*\).wav/\1.wav \1.mfc/" > codestr.scp
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\.wav/\.mfc/" > train.scp

# Convert WAVs to MFC feature vector files
HCopy -T 1 -C ../configuration/mfc-config -S codestr.scp

# Dictionary preparation (dict). Collect unique vocab terms and create a word list. Use an English lexicon provided by voxforge to associate phonemes and generate monophones0 list of monophones names (without sp, the name for short pause).
# See http://www.voxforge.org/home/dev/acousticmodels/linux/create/htkjulius/tutorial/data-prep/step-2
cat raw-audio/raw-sentences.txt | sed -E 's/[[:punct:]]/ /g' | tr '[:lower:]' '[:upper:]' | sed 's/^ //g' > vocab.txt
julia ../scripts/prompts2wlist.jl vocab.txt wlist
HDMan -A -D -T 1 -m -w wlist -n monophones0 -i -l dlog -g ../configuration/global.ded dict ../configuration/voxforge_lexicon_dict

# Create monophones1 which includes sp (short pause)
cp monophones0 monophones1
cat monophones1 | grep -v 'sp' > monophones0

# Use script mkphones0.led and dict to generate phonetical transcription phones0.mlf
HLEd -A -D -T 1 -l '*' -d dict -i phones0.mlf ../configuration/mkphones0.led words.mlf

cd ..

# Initialize proto model and output to hmm0 (compute mean and variance)
mkdir -p model/hmm0
HCompV -C configuration/hmm-config -f 0.01 -m -S train/train.scp -M model/hmm0 configuration/proto

# Clone initialized proto model for all phones from monophones0 list (hmm0/macros, hmm0/hmmdesf).
echo "" > model/hmm0/hmmdefs
head -n 3 model/hmm0/proto > model/hmm0/macros
cat model/hmm0/vFloors >> model/hmm0/macros
for w  in `cat train/monophones0`
do
 cat model/hmm0/proto | sed "s/proto/$w/g" | sed "1 d" | sed "1 d" | sed "1 d" >> model/hmm0/hmmdefs
done

# Train first three training cycles hmm1-hmm3 using Baum-Welch training on flat start monophones
mkdir -p model/hmm1
HERest -T 1 -C configuration/hmm-config -I train/phones0.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm0/macros -H model/hmm0/hmmdefs -M model/hmm1 train/monophones0

mkdir -p model/hmm2
HERest -T 1 -C configuration/hmm-config -I train/phones0.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm1/macros -H model/hmm1/hmmdefs -M model/hmm2 train/monophones0

mkdir -p model/hmm3
HERest -T 1 -C configuration/hmm-config -I train/phones0.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm2/macros -H model/hmm2/hmmdefs -M model/hmm3 train/monophones0

# Train hmm4-hmm5 with a silence model modification
# Copy hmm3 to hmm4
# In hmm4/hmmedefs, manually create one-state sp model with parameters of middle state of sil model
cp -r model/hmm3 model/hmm4
mkdir -p model/hmm5

cat model/hmm4/hmmdefs | grep -A5000 -m1 -e '~h "sil"' | sed 's/sil/sp/' | sed '/<STATE> 2/,/<STATE> 3/{//!d}' | sed '/<STATE> 4/,/<TRANSP> 5/{//!d}' | grep -v '<STATE> 2' | grep -v '<STATE> 4' | sed 's/<NUMSTATES> 5/<NUMSTATES> 3/' | sed 's/<STATE> 3/<STATE> 2/' | sed '/<TRANSP>/,//d' >> model/hmm4/hmmdefs
cat configuration/hmm4-transp.txt >> model/hmm4/hmmdefs

# Run HHEd to "tie" the sp state to the sil center state and output to hmm5 - tying means that one or more HMMs share the same set of parameters
HHEd -A -D -T 1 -H model/hmm4/macros -H model/hmm4/hmmdefs -M model/hmm5 configuration/sil.hed train/monophones1

# Generate new phonetic transcription with sp between words (phones1.mlf) but without "DE sp" command in script
HLEd -A -D -T 1 -l '*' -d train/dict -i train/phones1.mlf configuration/mkphones1.led train/words.mlf

# Next training cycles hmm6-hmm7 which contains sp model
mkdir -p model/hmm6
HERest -T 1 -C configuration/hmm-config -I train/phones1.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm5/macros -H model/hmm5/hmmdefs -M model/hmm6 train/monophones1

mkdir -p model/hmm7
HERest -T 1 -C configuration/hmm-config -I train/phones1.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm6/macros -H model/hmm6/hmmdefs -M model/hmm7 train/monophones1

# Extend dict by word for silence (dict2)
cp train/dict train/dict2
echo 'silence sil' >> train/dict2

# and run forced-alignment script
# The file aligned.mlf will have only sentences with good transcription
HVite -T 1 -l '*' -o SWT -b silence -C configuration/hmm-config -a -H model/hmm7/macros -H model/hmm7/hmmdefs -i train/aligned.mlf -m -t 250.0 -I train/words.mlf -S train/train.scp -y lab train/dict2 train/monophones1

# Run the last two training cycles hmm8-hmm9 on aligned data
mkdir -p model/hmm8
HERest -T 1 -C configuration/hmm-config -I train/aligned.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm7/macros -H model/hmm7/hmmdefs -M model/hmm8 train/monophones1

mkdir -p model/hmm9
HERest -T 1 -C configuration/hmm-config -I train/aligned.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm8/macros -H model/hmm8/hmmdefs -M model/hmm9 train/monophones1

# Create a triphone transcription on the aligned sentence data (wintri.mlf)
HLEd -A -D -T 1 -n train/triphones1 -l '*' -i train/wintri.mlf configuration/mktri.led train/aligned.mlf

# Create mktri.hed file
# When reestimating these new tied parameters the data from each of the original untied parameters is pooled so that a better estimate can be obtained. 
julia scripts/mktrihed.jl train/monophones1 train/triphones1 train/mktri.hed

# Run three more training cycles hmm10-hmm12 on triphone model
mkdir -p model/hmm10
HHEd -A -D -T 1 -H model/hmm9/macros -H model/hmm9/hmmdefs -M model/hmm10 train/mktri.hed train/monophones1 

mkdir -p model/hmm11
HERest  -A -D -T 1 -C configuration/hmm-config -I train/wintri.mlf -t 250.0 150.0 3000.0 -S train/train.scp -H model/hmm10/macros -H model/hmm10/hmmdefs -M model/hmm11 train/triphones1 

mkdir -p model/hmm12
HERest  -A -D -T 1 -C configuration/hmm-config -I train/wintri.mlf -t 250.0 150.0 3000.0 -s train/stats -S train/train.scp -H model/hmm11/macros -H model/hmm11/hmmdefs -M model/hmm12 train/triphones1

# Make a tied-state triphone dictionary (dict-tri)
HDMan -A -D -T 1 -b sp -n train/fulllist0 -g configuration/maketriphones.ded -l train/flog train/dict-tri train/dict2
# HDMan -A -D -T 1 -b sp -n train/fulllist0 -g configuration/maketriphones.ded -l train/flog train/dict-tri configuration/voxforge_lexicon_dict

# Append the contents of monophones0 to the beginning of to the fulllist0 file, and then to to remove any duplicate entries, and put the result in fulllist
julia scripts/fixfulllist.jl train/fulllist0 train/monophones0 train/fulllist

# Copy tree1.hed to tree.hed, append state clusters to tree.hed
cat configuration/tree1.hed > train/tree.hed
julia scripts/mkclscript.jl train/monophones0 train/tree.hed

# Run three more training cycles hmm13-hmm15
mkdir -p model/hmm13
cd train
HHEd -A -D -T 1 -H ../model/hmm12/macros -H ../model/hmm12/hmmdefs -M ../model/hmm13 tree.hed triphones1
cd ..

mkdir -p model/hmm14
HERest -A -D -T 1 -T 1 -C configuration/hmm-config -I train/wintri.mlf  -t 250.0 150.0 3000.0 -S train/train.scp -H model/hmm13/macros -H model/hmm13/hmmdefs -M model/hmm14 train/tiedlist

mkdir -p model/hmm15
HERest -A -D -T 1 -T 1 -C configuration/hmm-config -I train/wintri.mlf  -t 250.0 150.0 3000.0 -S train/train.scp -H model/hmm14/macros -H model/hmm14/hmmdefs -M model/hmm15 train/tiedlist

# Append "uw" state to end of hmmdefs
cat model/hmm9/hmmdefs | grep -A5000 -m1 -e '~h "uw"' | sed '/~h "sil"/,//d' >> model/hmm15/hmmdefs
