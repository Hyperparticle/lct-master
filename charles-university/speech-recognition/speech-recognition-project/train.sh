#!/usr/bin/env bash



cd train

# Clean up extraneous files
rm -f converted-audio/*.mfc
rm -f sentences.txt words.mlf codestr.scp train.scp vocab.txt wlist dlog dict monophones0 monophones1 phones0.mlf phones1.mlf dict2 aligned.mlf

# 2. Generate transcription file from sentences with following script (toMLF.prl, words.mlf)
cat raw-audio/raw-sentences.txt | sed -E 's/^.*\t//g' > sentences.txt
cat sentences.txt | sed -E 's/[[:punct:]]/ /g' | grep . | perl ../scripts/toMLF.prl > words.mlf

# 3. Generate conversion and MFC list (codestr.scp, train.scp)
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\(.*\).wav/\1.wav \1.mfc/" > codestr.scp
ls -1 "$PWD"'/converted-audio/'*.wav | sed 's/ /\\ /g' | sed "s/\.wav/\.mfc/" > train.scp

# 4. Convert WAVs to MFC feature vector files
HCopy -T 1 -C ../configuration/mfc-config -S codestr.scp

# 5. Dictionary preparation (dict). Collect unique vocab terms and create a word list. Use an English lexicon provided by voxforge to associate phonemes and generate monophones0 list of monophones names (without sp, the name for short pause).
# See http://www.voxforge.org/home/dev/acousticmodels/linux/create/htkjulius/tutorial/data-prep/step-2
cat raw-audio/raw-sentences.txt | sed -E 's/[[:punct:]]/ /g' | tr '[:lower:]' '[:upper:]' | sed 's/^ //g' > vocab.txt
julia ../scripts/prompts2wlist.jl vocab.txt wlist
HDMan -A -D -T 1 -m -w wlist -n monophones0 -i -l dlog -g ../configuration/global.ded dict ../configuration/voxforge_lexicon_dict

# Create monophones1 which includes sp
cp monophones0 monophones1
cat monophones1 | grep -v 'sp' > monophones0

# 6. Create script mkphones0.led and generate phonetical transcription by command
HLEd -A -D -T 1 -l '*' -d dict -i phones0.mlf ../configuration/mkphones0.led words.mlf

cd ..

# 7. hmm0 (compute mean and variance)
# Initialize proto model
mkdir -p model/hmm0
HCompV -C configuration/hmm-config -f 0.01 -m -S train/train.scp -M model/hmm0 configuration/proto

# 8. Clone initialized proto model for all phones from monophones0 list (hmm0/macros, hmm0/hmmdesf).
echo "" > model/hmm0/hmmdefs
head -n 3 model/hmm0/proto > model/hmm0/macros
cat model/hmm0/vFloors >> model/hmm0/macros
for w  in `cat train/monophones0`
do
 cat model/hmm0/proto | sed "s/proto/$w/g" | sed "1 d" | sed "1 d" | sed "1 d" >> model/hmm0/hmmdefs
done

# cat hmm0/proto | tail -n +5 > hmm0/proto-tail
# cat converted/monophones0 | sed 's/.*/~h "&"/' | sed '/.*/r hmm0/proto-tail' > hmm0/hmmdefs
# echo -e "$(cat hmm0/proto | head -n 3)\n$(cat hmm0/vFloors)" > hmm0/macros

# 9. hmm1-hmm3 (flat start monophones)
#  First three training cycles (Baum-Welch training)
mkdir -p model/hmm1
HERest -T 1 -C configuration/hmm-config -I train/phones0.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm0/macros -H model/hmm0/hmmdefs -M model/hmm1 train/monophones0

mkdir -p model/hmm2
HERest -T 1 -C configuration/hmm-config -I train/phones0.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm1/macros -H model/hmm1/hmmdefs -M model/hmm2 train/monophones0

mkdir -p model/hmm3
HERest -T 1 -C configuration/hmm-config -I train/phones0.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm2/macros -H model/hmm2/hmmdefs -M model/hmm3 train/monophones0

# 10. hmm4-hmm5
# Silence model modification
# In hmm4/hmmedefs create manualy one-state sp model with parameters of middle state of sil model.
cp -r model/hmm3 model/hmm4
mkdir -p model/hmm5

cat model/hmm4/hmmdefs | grep -A5000 -m1 -e '~h "sil"' | sed 's/sil/sp/' | sed '/<STATE> 2/,/<STATE> 3/{//!d}' | sed '/<STATE> 4/,/<TRANSP> 5/{//!d}' | grep -v '<STATE> 2' | grep -v '<STATE> 4' | sed 's/<NUMSTATES> 5/<NUMSTATES> 3/' | sed 's/<STATE> 3/<STATE> 2/' | sed '/<TRANSP>/,//d' >> model/hmm4/hmmdefs
cat configuration/hmm4-transp.txt >> model/hmm4/hmmdefs

HHEd -A -D -T 1 -H model/hmm4/macros -H model/hmm4/hmmdefs -M model/hmm5 configuration/sil.hed train/monophones1

# Generate new phonetic transcription with sp between words (phones1.mlf) but without "DE sp" command in script.
HLEd -A -D -T 1 -l '*' -d train/dict -i train/phones1.mlf configuration/mkphones1.led train/words.mlf

# 11. hmm6-hmm7
# Next training cycles
mkdir -p model/hmm6
HERest -T 1 -C configuration/hmm-config -I train/phones1.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm5/macros -H model/hmm5/hmmdefs -M model/hmm6 train/monophones1

mkdir -p model/hmm7
HERest -T 1 -C configuration/hmm-config -I train/phones1.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm6/macros -H model/hmm6/hmmdefs -M model/hmm7 train/monophones1

# Extend dict by word for silence: 
cp train/dict train/dict2
echo 'silence sil' >> train/dict2

# and run forced-alignment script:
# In file aligned.mlf will be only sentences with good transcription.
HVite -T 1 -l '*' -o SWT -b silence -C configuration/hmm-config -a -H model/hmm7/macros -H model/hmm7/hmmdefs -i train/aligned.mlf -m -t 250.0 -I train/words.mlf -S train/train.scp -y lab train/dict2 train/monophones1

# 16. Last two training cycles:
# hmm8-9
mkdir -p model/hmm8
HERest -T 1 -C configuration/hmm-config -I train/aligned.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm7/macros -H model/hmm7/hmmdefs -M model/hmm8 train/monophones1

mkdir -p model/hmm9
HERest -T 1 -C configuration/hmm-config -I train/aligned.mlf -t 250.0 150.0 1000.0 -S train/train.scp -H model/hmm8/macros -H model/hmm8/hmmdefs -M model/hmm9 train/monophones1
