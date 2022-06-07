#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

Thanos_original_work_dir=/home/mgilani/data/Exact/

#python train.py data_dir=${Thanos_original_work_dir} experiment=exact_vicreg.yaml datamodule.num_workers=8

# finetune
python train.py -m data_dir=${Thanos_original_work_dir} experiment=exact_finetune.yaml datamodule.num_workers=8 seed=13 datamodule.split_randomstate=5,26

#python train.py trainer.max_epochs=10 loggerv=cs
