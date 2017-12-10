#!/usr/bin/env bash

#hardware_end=142
#sim_end=120

print_hardware_result() {
    k=$1; s=$2
    result=$(./matrix ${k} ${s})

    n=$(echo ${result} | cut -f 1 -d ",")
    frac=$((${n} * (${n} - 1) / 2))
    swap=$(echo ${result} | cut -f 2 -d "," | awk '{print $1 / '"${frac}"'}')

    echo "${result},${swap}"
}

echo "Simple Test"
echo "n,simple_total,simple"
for k in {54..143}
do
    print_hardware_result ${k} -s
done
echo

echo "Recursive Test"
echo "n,recursive_total,recursive"
for k in {54..143}
do
    print_hardware_result ${k}
done
echo



ceil() {
  echo "define ceil (x) {if (x<0) {return x/1} \
        else {if (scale(x)==0) {return x} \
        else {return x/1 + 1 }}} ; ceil($1)" | bc
}

print_sim_result() {
    k=$1; B=$2; C=$3; s=$4

    result=$(./matrix-print ${k} ${s} | ./cachesim ${B} ${C} | sed -n 3p | cut -f 2 -d " ")

    n_frac=$(echo ${k} | awk '{print 2.0^($1 / 9.0)}')
    n=$(ceil ${n_frac})
    frac=$((${n} * (${n} - 1) / 2))

    swap=$(echo ${result} | awk '{print $1 / '"${frac}"'}')

    echo "${B},${C},${k},${n},${result},${swap}"
}

echo "Cache Test (Simple)"
echo "B,C,k,n,simple_total,simple"
for i in 64,64 64,1024 64,4096 512,512 4096,64; do
    IFS=',' read B C <<< "${i}"
    for k in {54..126}
    do
        print_sim_result ${k} ${B} ${C} -s
    done
done
echo

echo "Cache Test (Recursive)"
echo "B,C,k,n,recursive_total,recursive"
for i in 64,64 64,1024 64,4096 512,512 4096,64; do
    IFS=',' read B C <<< "${i}"
    for k in {54..126}
    do
        print_sim_result ${k} ${B} ${C}
    done
done
echo
