#!/usr/bin/env bash

id=48

./splaygen -s ${id} -t $1 | ./splaytree
