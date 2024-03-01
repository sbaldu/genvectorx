#!/bin/bash

# $1: platform (i.e. gpu model)
# $2: device index

script="$(dirname "$(readlink -f -- "$0")")/plots.py"
path="$(dirname "$(readlink -f -- "$0")")/"
nrun=3
environments="cuda"
platform=$1
output="$(date "+%y%m%d-%H%M%S")"
declare -a sycl_mms=("buf" "ptr")
declare -a tests=("InvariantMasses" "Boost")


for e in "$environments"
do      
    for s in ${sycl_mms[@]}
    do
        for t in ${tests[@]}
        do
            #echo "ACPP_VISIBILITY_MASK=$e ONEAPI_DEVICE_SELECTOR=$e:0 python $script ${path} ${t} ${platform} ${e} ${s} $nrun $output"
            ACPP_VISIBILITY_MASK=$e CUDA_VISIBLE_DEVICES=$2 ONEAPI_DEVICE_SELECTOR=$e:$2 python3 $script ${path} ${t} ${platform} ${e} ${s} $nrun $output
        done
    done
done

