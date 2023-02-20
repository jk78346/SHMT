#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_host_build"

cd "${GITTOP}/${BUILD_DIR}"
iter=100
baseline_mode="gpu"

for app_name in  "dwt_2d"  #"mean_2d" "sobel_2d" "laplacian_2d" "fft_2d" "dct8x8_2d" "srad_2d" "hotspot_2d"
do
    for problem_size in 1024 2048 4096 8192
    do
        # iter baseline
        sudo ./gpgtpu ${app_name} ${problem_size} ${problem_size} ${iter} ${baseline_mode} gpu
        for block_size in 512 1024 2048
        do
            for proposed_mode in "gt_c-ns-sdev" "gt_c-nr-sdev" "gt_c-ns-range" "gt_c-nr-range" 
            do
                for num in 5
                do
                    sudo ./gpgtpu ${app_name} ${problem_size} ${block_size} ${iter} ${baseline_mode} ${proposed_mode} ../data/super${num}.png
                done
            done
        done
    done
done
