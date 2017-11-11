#!/usr/bin/env bash

id=48

echo "Random Test"
./heapgen -s ${id} -r | ./fibheap
echo

echo "Random Test (naive)"
./heapgen -s ${id} -r | ./fibheap -n
echo

echo "Biased Test"
./heapgen -s ${id} -b | ./fibheap
echo

echo "Biased Test (naive)"
./heapgen -s ${id} -b | ./fibheap -n
echo

echo "Special Test"
./heapgen -s ${id} -x | ./fibheap
echo

echo "Special Test (naive)"
./heapgen -s ${id} -x | ./fibheap -n
echo
