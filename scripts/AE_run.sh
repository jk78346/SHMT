#!/bin/sh

#GITTOP="$(git rev-parse --show-toplevel 2>&1)"

iter=100
baseline_mode="gpu"

for p in 128 #2 4 8 16 32 64 128
do
    for app_name in  "mean_2d" "sobel_2d" "laplacian_2d" "fft_2d" "dct8x8_2d" "srad_2d" "hotspot_2d" "dwt_2d" "blackscholes_2d" "histogram_2d"
    do
        for problem_size in 8192
        do
            for block_size in 2048
            do
                for proposed_mode in "gt_c"

                do
                    sudo ./gpgtpu ${app_name} ${problem_size} ${block_size} ${iter} ${baseline_mode} ${proposed_mode} ${p} ../data/super5.png
                done
            done
        done
    done
done
