#!/usr/bin/env bash

# Converts .m4a files to .wav
# Usage: $ convert2wav.sh <input-directory> <output-directory>

# Convert recorded sentences to .wav using avconv (or ffmpeg)
for i in $1/*.m4a
do
  # ffmpeg -i "$i" "${i%.*}.wav"
  avconv -i "$i" "${i%.*}.wav"
done

mv $1/*.wav $2

# Rename to S001, S002, ...
for i in $2/*.wav
do
  rename=$(echo $i | sed -E 's/^.*\((.*)\).*$/\1/g' | xargs printf 'S%03d')
  mv "${i}" "$2/${rename}.wav"
done
