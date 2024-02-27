#!/bin/bash

script="$(dirname "$(readlink -f -- "$0")")/plots.py"
path="$(dirname "$(readlink -f -- "$0")")/"
nrun=1
environments="cuda"
platform="rtx3060"
output="$(date "+%y%m%d-%H%M%S")"
declare -a sycl_mms=("buf" "ptr")
declare -a tests=("InvariantMasses" "Boost")

# python /home/mdessole/Projects/GenVectorX/genvectorx/plots/plots.py Boost cpu 3
# python /home/mdessole/Projects/GenVectorX/genvectorx/plots/plots.py InvariantMasses cpu 3

# ACPP_VISIBILITY_MASK=ocl ONEAPI_DEVICE_SELECTOR=opencl:0 python /home/mdessole/Projects/GenVectorX/genvectorx/plots/plots.py Boost fpga_$1 3
# ACPP_VISIBILITY_MASK=ocl ONEAPI_DEVICE_SELECTOR=opencl:1 python /home/mdessole/Projects/GenVectorX/genvectorx/plots/plots.py Boost ocl_$1 3
#ACPP_VISIBILITY_MASK=cuda ONEAPI_DEVICE_SELECTOR=cuda:0 python ./plots.py Boost cuda_$1 3

# ACPP_VISIBILITY_MASK=ocl ONEAPI_DEVICE_SELECTOR=opencl:1 python /home/mdessole/Projects/GenVectorX/genvectorx/plots/plots.py InvariantMasses ocl_$1 3
# ACPP_VISIBILITY_MASK=ocl ONEAPI_DEVICE_SELECTOR=opencl:0 python /home/mdessole/Projects/GenVectorX/genvectorx/plots/plots.py InvariantMasses fpga_$1 3
# ACPP_VISIBILITY_MASK=cuda ONEAPI_DEVICE_SELECTOR=cuda:0 python

for e in "$environments"
do      
    for s in ${sycl_mms[@]}
    do
        for t in ${tests[@]}
        do
            #echo "ACPP_VISIBILITY_MASK=$e ONEAPI_DEVICE_SELECTOR=$e:0 python $script ${path} ${t} ${platform} ${e} ${s} $nrun $output"
            ACPP_VISIBILITY_MASK=$e ONEAPI_DEVICE_SELECTOR=$e:0 python $script ${path} ${t} ${platform} ${e} ${s} $nrun $output
        done
    done
done

