#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['levir', 'whu']
# exp: just for specifying the 'save_path'
# split: ['5%', '10%', '20%', '40%']
dataset='LEVIR-256'
method='FL-SCD'
exp='FL-SCD_1'
split='40'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split


config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split


mkdir -p $save_path

 CUDA_VISIBLE_DEVICES=3 nohup torchrun --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=1456  \
    /home/zhanyikuan/FL-SCD/unimatch_SSM_FRE.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save_path $save_path --port 1456 &

