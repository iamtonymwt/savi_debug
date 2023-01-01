#!/bin/bash
#echo Fold:[[1,3],[2,5],[4,8],[6,7]]
#echo Exp starting! fold order: 0,1,2,3

#python -m savi.main --wandb --group semi_nbb --exp exp27_2 --fold 2 --resume_from base/model_2.pt;
#python -m savi.main --fold 0 --resume_from base/model_0.pt;
#python -m savi.main --fold 2 --resume_from 10000.pt;
#python -m savi.main --eval --fold 0 --resume_from base/model_0.pt;

#---------------------------------------------------------------------------

#python -m savi.main --wandb --group semi_nbb --exp exp33_7 --fold 7 --resume_encoder 18base/model_7.pt ;
#python -m savi.main --fold 2 --resume_encoder 10base/model_2.pt --resume_attention 10000.pt;
#python -m savi.main --fold 7 --resume_encoder 18base/model_7.pt --eval

#---------------------------------------------------------------------------

#python -m savi.main --wandb --group semi_resnet34 --exp exp34_7 --fold 7 --resume_encoder deeplabv3_resnet34_18base/model_7.pt ;
#python -m savi.main --fold 7 --resume_encoder deeplabv3_resnet34_18base/model_7.pt ;
#python -m savi.main --fold 7 --resume_attention 10000.pt --visual;
#python -m savi.main --wandb --group semi_resnet34 --exp exp38_7 --fold 7 --resume_attention 10000.pt;
#python -m savi.main --group semi_resnet34 --exp exp37_7 --fold 7 --visual --resume_from experiments/exp37_7/snapshots/79.pt

#----------------------------------------------------------------------------
#python -m savi.main --wandb --group semi_resnet34 --exp exp40_7 --fold 7 --resume_attention 10000.pt;
#python -m savi.main  --fold 0 --eval --resume_attention 10000.pt;

#----------------------------------------------------------------------------

#python -m savi.main --wandb --group savi++ --exp exp42_0_4 --fold 0 ;
python -m savi.main  --fold 0 ;
#python -m savi.main --fold 0 --resume_encoder deeplabv3_resnet34_4base/model_0.pt;
#python -m savi.main --wandb --group savi++ --exp exp44_0_4 --fold 0 --resume_encoder deeplabv3_resnet34_4base/model_0.pt;
#python -m savi.main --wandb --group savi++ --exp exp44_0_4 --fold 0


