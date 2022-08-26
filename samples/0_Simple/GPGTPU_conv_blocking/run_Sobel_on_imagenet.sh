#!/bin/sh

in_root="/nfshome/khsu037/ILSVRC/Data/"
DET="DET/"
img_size=2048

for seg in "train/" "test/" "val/"
do
    in_dir="${in_root}${DET}${seg}"
    resized_in_dir="${in_root}/Sobel_${img_size}/in/${seg}"
    out_dir="${in_root}/Sobel_${img_size}/out/${seg}"
    mkdir -p ${resized_in_dir}
    mkdir -p ${out_dir}
    ./Sobel ${in_dir} ${resized_in_dir} ${out_dir} ${img_size}
done

