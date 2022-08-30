#!/bin/bash

#SBATCH -J run
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --partition=t4v1,t4v2,rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90

#aug=baseline_augs
#
#/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
#    datamodule/augmentations=$aug \
#    logger.wandb.group=${aug}_weaker \
#    trainer.pretrain.min_epochs=200 \
#    name=${aug}_200epoch \
#    trainer.pretrain.resume_from_checkpoint=/h/pwilson/projects/SSLMicroUltrasound/checkpoints/crp_plus_us_augs_sd3_8102277/last.ckpt

#/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
#    model.pretrain.backbone=resnet18 \
#    experiment=pretrain_both_centers \
#    trainer.pretrain.min_epochs=1000 \

#/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
#    datamodule.self_supervised.needle_region_only=True \
#    datamodule/augmentations=crops_plus_ultrasound_augs \
#    seed=$seed \
#    name=vicreg_needle_region_only_false_sd${seed} \
#    logger.wandb.group=needle_region_only
#
#/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
#    experiment=byol

#/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
#    experiment=fully_supervised
#


#name=sup_imnet-init_us-augs
#
#/h/pwilson/anaconda3/envs/exact/bin/python train.py \
#    experiment=fully_supervised \
#    datamodule.resample_train_val_seed=$seed0 \
#    seed=$seed1 \
#    logger.wandb.group=${name} \
#    name=${name} \
#    model.backbone=resnet18_imagenet  #resnet18_imagenet, #resnet10


name=vicreg_pre-D-uva-prostate_lineval-D-uva-needle_resnet16_crop_plus_us

/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
    datamodule/self_supervised=self_supervised \
    datamodule.self_supervised.needle_region_only=False \
    datamodule.self_supervised.resample_train_val_seed=${seed0} \
    datamodule/supervised=train_uva_test_both \
    datamodule/augmentations=crops_plus_ultrasound_augs \
    datamodule.supervised.resample_train_val_seed=${seed0} \
    seed=${seed1} \
    name=${name} \
    logger.wandb.group=${name} \
    model.pretrain.backbone=resnet18 \
    trainer.pretrain.resume_from_checkpoint=null


#/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
#    datamodule.self_supervised.needle_region_only=False \
#    datamodule.self_supervised.resample_train_val_seed=${seed0} \
#    datamodule/supervised=train_uva_test_both \
#    datamodule.supervised.resample_train_val_seed=${seed0} \
#    seed=${seed1} \
#    name=${name} \
#    logger.wandb.group=${name} 