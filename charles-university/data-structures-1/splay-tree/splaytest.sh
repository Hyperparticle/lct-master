#!/usr/bin/env bash

id=48
end= #-l

T=(10 100 1000 10000 100000 1000000)

echo "Uniform subset test"
for i in "${T[@]}"
do
    echo ${i}
    ./splaygen -s ${id} -t ${i} ${end} | ./splaytree
done

echo

echo "Uniform subset test (naive)"
for i in "${T[@]}"
do
    echo ${i}
    ./splaygen -s ${id} -t ${i} ${end} | ./splaytree -n
done

echo

echo "Sequential test"
./splaygen -s ${id} -b ${end} | ./splaytree

echo

echo "Sequential test (naive)"
./splaygen -s ${id} -b ${end} | ./splaytree -n
