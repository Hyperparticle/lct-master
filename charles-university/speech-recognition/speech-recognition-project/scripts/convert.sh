#!/usr/bin/env bash

# Convert recorded training and test sentences from .m4a to .wav
bash scripts/convert2wav.sh train/raw-audio train/converted-audio
bash scripts/convert2wav.sh test/raw-audio test/converted-audio

