#!/usr/bin/env bash

id=48

if [ "$#" -ne 1 ]; then
    echo "Usage: splay.sh [size]"
    echo "size - integer size of random subset to be generated"
    exit 1
fi

./splaygen -s ${id} -t $1 | ./splaytree
