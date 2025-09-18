#!/bin/bash

idset="imagenet200"
arch="resnet18"
# Loop through each path
for path in \
    "imagenet200_v1_s1" \
    ; do
    mod_path="models/${path}/best.ckpt"
    save_path="${path}"
    python extract_feature.py \
    --id_name $idset \
    --arch $arch \
    --ckpt-path $mod_path \
    --save-path $save_path \
    --num_workers 0 # \
    # --skip_train
done

idset="imagenet200"
arch="resnet18"
# Loop through each path
for path in \
    "imagenet200_ls_s1" \
    "imagenet200_mixup_s1" \
    "imagenet200_mixup_ls_s1" \
    "imagenet200_v2_s1" \
    "imagenet200_augrevise_s1" \
    ; do
    mod_path="models/${path}/model_best.pth"
    save_path="${path}"
    python extract_feature.py \
    --id_name $idset \
    --arch $arch \
    --ckpt-path $mod_path \
    --save-path $save_path \
    --num_workers 0 # \
    # --skip_train
done




idset="imagenet"
arch="resnet50"
# Loop through each path
for path in \
    "imagenet_resnet50_augrev_m100" \
    ; do
    mod_path="models/${path}/model_best.pth"
    save_path="${path}"
    python extract_feature.py \
    --id_name $idset \
    --arch $arch \
    --ckpt-path $mod_path \
    --save-path $save_path \
    --num_workers 0 # \
    # --skip_train
done

idset="imagenet"
arch="swin-t"
# Loop through each path
for path in \
    "imagenet_swin-t_augrev_m10" \
    ; do
    mod_path="models/${path}/model_best.pth"
    save_path="${path}"
    python extract_feature.py \
    --id_name $idset \
    --arch $arch \
    --ckpt-path $mod_path \
    --save-path $save_path \
    --num_workers 0 # \
    # --skip_train
done
