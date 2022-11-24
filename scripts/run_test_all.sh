#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_host_build"

cd "${GITTOP}/${BUILD_DIR}"
iter=1
baseline_mode="cpu"
log_file_path="../log/record.csv"

for app_name in "sobel_2d"
do
    for problem_size in 2048 4096 8192
    do
        for block_size in 512 1024 2048
        do
            if [ ${problem_size} -gt ${block_size} ]
            then
                for proposed_mode in "cpu_p" "gpu_p" "tpu_p" "all_p" "gtr_p" "ctr_p" "cgr_p"
                do
                    sudo ./gpgtpu ${app_name} ${problem_size} ${block_size} ${iter} ${baseline_mode} ${proposed_mode} ${log_file_path}
                done
            fi
        done
    done
done
