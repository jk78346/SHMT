#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_host_build"

cd "${GITTOP}/${BUILD_DIR}"
iter=100
baseline_mode="gpu"

for app_name in "mean_2d" "sobel_2d" "laplacian_2d" "fft_2d" "dct8x8_2d" "srad_2d" "hotspot_2d" "dwt_2d" "blackscholes_2d" "histogram_2d"
do
    for problem_size in 64 128 256 512 1024 2048 4096 8192 16384
    do
        for block_size in 64 128 256 512 1024 2048
        do
            for proposed_mode in "gt_c-ns-dev" #"gt_c-homo" #"gt_c-oracle" #"gt_c-ns-sdev" "gt_c-nr-sdev" "gt_c-ns-range" "gt_c-nr-range" "gt_c-homo"
            do
                for p in 128 #0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
                do
                    for num in 5 #$(seq -w 0001 0900)
                    do
                        sudo ./gpgtpu ${app_name} ${problem_size} ${block_size} ${iter} ${baseline_mode} ${proposed_mode} ${p} ../data/super${num}.png
                    done
                done
            done
        done
    done
done
