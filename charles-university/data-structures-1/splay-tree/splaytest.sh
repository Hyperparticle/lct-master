#!/usr/bin/env bash

id=48

T=(10 100 1000 10000 100000 1000000)

echo "Uniform subset test"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -t ${i} | ./splaytree)
done

echo

echo "Uniform subset test (naive)"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -t ${i} | ./splaytree -n)
done

echo

echo "Sequential test"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -b | ./splaytree)
done

echo

echo "Sequential test (naive)"
for i in "${T[@]}"
do
    echo ${i},$(./splaygen -s ${id} -b | ./splaytree -n)
done
