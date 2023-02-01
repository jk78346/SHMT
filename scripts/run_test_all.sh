#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_host_build"

cd "${GITTOP}/${BUILD_DIR}"
iter=100
baseline_mode="gpu"

for app_name in "fft_2d" #"mean_2d" "sobel_2d" "laplacian_2d" "fft_2d" "dct8x8_2d"
do
    for problem_size in 1024 2048 4096 8192
    do
        # iter baseline
        sudo ./gpgtpu ${app_name} ${problem_size} ${problem_size} ${iter} ${baseline_mode} gpu
        
        for block_size in 512 1024 2048
        do
            for proposed_mode in "t_p" "gt_s" "gt_b" "gt_c" 
            do
                sudo ./gpgtpu ${app_name} ${problem_size} ${block_size} ${iter} ${baseline_mode} ${proposed_mode}
            done
        done
    done
done
