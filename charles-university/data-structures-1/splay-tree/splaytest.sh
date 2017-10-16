#!/usr/bin/env bash

id=48

T=(10 100 1000 10000 100000 1000000)

echo "Uniform subset test"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -t ${i} | ./splaytree)
done

echo "Sequential test"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -t ${i} -b | ./splaytree)
done


T=(100 10000 1000000)

echo "Sequential test"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -t ${i} -b | ./splaytree)
done

echo "Sequential test (naive)"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -t ${i} -b | ./splaytree -n)
done
