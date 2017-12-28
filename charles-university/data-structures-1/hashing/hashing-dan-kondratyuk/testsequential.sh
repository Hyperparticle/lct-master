#!/usr/bin/env bash

./hashing st | ./stats.py | tee docs/sequential-tabulation.csv
./hashing sm | ./stats.py | tee docs/sequential-multiply-shift.csv
