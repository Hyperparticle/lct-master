#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TAG=$DIR/brill/Bin_and_Data
TAGMAKE=$DIR/brill
UTIL=$DIR/brill/Utilities
DATA=$DIR/data

TOP=300

cd $DATA

# rm TAGGED-CORPUS  TAGGED-CORPUS-2 TAGGED-CORPUS-ENTIRE UNTAGGED-CORPUS  UNTAGGED-CORPUS-2

rm -f BIGBIGRAMLIST  BIGWORDLIST  DUMMY-TAGGED-CORPUS  FINAL.LEXICON  LEXRULEOUTFILE  SMALLWORDTAGLIST TRAINING.LEXICON  

# cat texten2.ptg | tr '\n' ' ' > TAGGED-CORPUS
# cat TAGGED-CORPUS | perl $UTIL/tagged-to-untagged.prl > UNTAGGED-CORPUS

cat UNTAGGED-CORPUS | perl $UTIL/wordlist-make.prl | sort +1 -rn | awk '{ print $1}' > BIGWORDLIST
cat TAGGED-CORPUS | perl $UTIL/word-tag-count.prl | sort +2 -rn > SMALLWORDTAGLIST 
cat UNTAGGED-CORPUS | perl $UTIL/bigram-generate.prl | awk '{ print $1,$2}' > BIGBIGRAMLIST
# perl $DIR/brill/Learner_Code/unknown-lexical-learn.prl BIGWORDLIST SMALLWORDTAGLIST BIGBIGRAMLIST $TOP LEXRULEOUTFILE
cat TAGGED-CORPUS | perl $UTIL/make-restricted-lexicon.prl > TRAINING.LEXICON
cat TAGGED-CORPUS-ENTIRE | perl $UTIL/make-restricted-lexicon.prl > FINAL.LEXICON
# cat TAGGED-CORPUS-2 | perl $UTIL/tagged-to-untagged.prl > UNTAGGED-CORPUS-2

cd $TAGMAKE

make 2> /dev/null

cd $TAG

./tagger TRAINING.LEXICON UNTAGGED-CORPUS-2 BIGBIGRAMLIST \
   LEXRULEOUTFILE /dev/null -w BIGWORDLIST  \
   -i DUMMY-TAGGED-CORPUS > /dev/null
