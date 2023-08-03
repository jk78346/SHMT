#!/bin/sh

iter=1
baseline_mode="gpu"

for problem_size in 8192
do
    for proposed_mode in "gt_c-ks" # QAWS-KS
    do
        sudo ./gpgtpu "mean_2d" ${problem_size} 512 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "sobel_2d" ${problem_size} 512 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "laplacian_2d" ${problem_size} 2048 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "fft_2d" ${problem_size} 1024 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "dct8x8_2d" ${problem_size} 1024 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "srad_2d" ${problem_size} 512 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "hotspot_2d" ${problem_size} 2048 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "dwt_2d" ${problem_size} 2048 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "blackscholes_2d" ${problem_size} 256 ${iter} ${baseline_mode} ${proposed_mode}  
        sudo ./gpgtpu "histogram_2d" ${problem_size} 2048 ${iter} ${baseline_mode} ${proposed_mode}  
    done
done
