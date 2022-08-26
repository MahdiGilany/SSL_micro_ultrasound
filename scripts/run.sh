export HYDRA_FULL_ERROR=1

/h/pwilson/anaconda3/envs/exact/bin/python train_combined.py \
    experiment=pretrain_both_centers \
    trainer.pretrain.resume_from_checkpoint=/h/pwilson/projects/SSLMicroUltrasound/checkpoints/pretrain_both_centers_8097042/last.ckpt
    trainer.pretrain.max_epochs = 