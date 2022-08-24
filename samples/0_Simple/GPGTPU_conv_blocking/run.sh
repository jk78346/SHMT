#!/bin/sh

iter=1
p=0.5
baseline_mode=0
input_data_mode=4 # 0: uniform, 1: normal, 2: Hankel, 3: Frank matrix, 4: read image 5: read mtx file 6: markov text generation
scale=261000


for size in 4096
do
  for blk_size in 2048 #$ 1024 512
  do
    if [ ${blk_size} -eq 1024 ]
    then
      scale=100000
    fi
    if [ ${blk_size} -eq 2048 ]
    then
      scale=142000 #193000 #142000
    fi
    if [ ${size} -gt ${blk_size} ] || [ ${size} -eq ${blk_size} ]
    then
      for mode in 2
      do
        ./conv ${input_data_mode} ${size} ${iter} ${scale} ${baseline_mode} ${mode} ${blk_size} ${p} 2>&1 | tee -a ./log/conv_run_record_${input_data_mode}_${size}_${iter}_${scale}_${baseline_mode}_${mode}_${blk_size}_${p}.txt
      done
    fi
  done
done
