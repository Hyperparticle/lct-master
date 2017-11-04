#!/usr/bin/env bash

id=48

if [ "$#" -ne 1 ]; then
    echo "Usage: fibheaptest.sh [test]"
    echo "[test] - -r (random test), -b (biased test), or -s (special test)"
    exit 1
fi

./heapgen -s ${id} $1 | ./fibheap
