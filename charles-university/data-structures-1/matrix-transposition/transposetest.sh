#!/usr/bin/env bash

echo "Simple Test"
for k in {54..142}
do
    ./matrix ${k} -s
done
echo

echo "Standard Test"
for k in {54..142}
do
    ./matrix ${k}
done
echo

echo "Cache Test (Simple)"
for i in 64,64 64,1024 64,4096 512,512 4096,64; do
    IFS=',' read B C <<< "${i}"
    for k in {54..120}
    do
        result=$(./matrix-print ${k} -s | ./cachesim ${B} ${C} | sed -n 3p | cut -f 2 -d " ")
        echo "${B},${C},${k},${result}"
    done
done
echo

echo "Cache Test (Standard)"
for i in 64,64 64,1024 64,4096 512,512 4096,64; do
    IFS=',' read B C <<< "${i}"
    for k in {54..120}
    do
        result=$(./matrix-print ${k} | ./cachesim ${B} ${C} | sed -n 3p | cut -f 2 -d " ")
        echo "${B},${C},${k},${result}"
    done
done
echo
