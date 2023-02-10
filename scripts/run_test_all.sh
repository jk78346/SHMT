#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_host_build"

cd "${GITTOP}/${BUILD_DIR}"
iter=100
baseline_mode="gpu"

for app_name in "sobel_2d" #"fft_2d" "dct8x8_2d" "hotspot_2d" "srad_2d"
do
    for problem_size in 4096 
    do
        # iter baseline
        #sudo ./gpgtpu ${app_name} ${problem_size} ${problem_size} ${iter} ${baseline_mode} gpu
        block_size=512
        for ratio in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1
        do
            for proposed_mode in "gt_c-oracle" "gt_c-nr" #"gt_c-oracle" "gt_c" "gt_c-ns" 
            do
                for num in 5
                do
                    sudo ./gpgtpu ${app_name} ${problem_size} ${block_size} ${iter} ${baseline_mode} ${proposed_mode} ${ratio} ../data/super${num}.png
                done
            done
        done
    done
done
