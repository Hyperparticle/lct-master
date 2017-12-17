#!/usr/bin/env bash

./hashing rct | ./avg.py | tee docs/cuckoo-tabulation.csv
./hashing rcm | ./avg.py | tee docs/cuckoo-multiply-shift.csv
./hashing rlt | ./avg.py | tee docs/linear-probing-tabulation.csv
./hashing rlm | ./avg.py | tee docs/linear-probing-multiply-shift.csv
./hashing rln | ./avg.py | tee docs/linear-probing-naive-modulo.csv

./hashing st | ./stats.py | tee docs/sequential-tabulation.csv
./hashing sm | ./stats.py | tee docs/sequential-multiply-shift.csv
