_base_ = [
    #'mmpretrain::_base_/datasets/imagenet_bs64_swin_256.py',
    'mmpretrain::_base_/default_runtime.py',
    #'mmpretrain::_base_/models/vit-base-p16.py',
    #'mmpretrain::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    ##'mmpretrain::vision_transformer/vit-base-p16_32xb128-mae_in1k.py',
    #'./mmbase_cfg.py',
]
import sys
sys.path.append('./')
from  mmbase_cfg import *
custom_imports = dict(imports='base_model')
gapsize = None
vittype = 3
in_channels = 1024#DIM_FEAT if not gapsize else DIM_FEAT*gapsize*gapsize

model.update(
    #image_size=256, patch_size=16, dim=768, depth=6, heads=8, mlp_dim = 512, pool = 'mean', dim_head = 64,init_cfg=None)
    backbone=dict(#_delete_=True,
        type='CI2PVit', img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM_FEAT, depth=VIT_DEPTH, vitBackbone=ViT_Backbone, 
        extCfg=dict(
            IMG_NORMALIZATION=IMG_NORMALIZATION,
            ci2pDownSample=False, 
            ci2pDimout=192,
            #patchFlatten=1, 
            #freezeViT=True,
            cnnPlusMergeIndex=3,
            
        ),
        #init_cfg=dict(
        #    type='Pretrained',
        #    prefix='backbone.',
        #    checkpoint='../../checkpoint/ci2pvit_imagenet/imagenet_epoch_385.pth',
        #),
        vittype=vittype,
		ConvPlusCfg = dict(
            plusCNNModelCfg=dict(type='mmpretrain.EfficientNet', arch='b1', out_indices=(6, 5, 4, 3, 2, 1, 0)),
            plusCNNMergeStep = ((4, 192, 192), (5, 768, 768), (6, 768, in_channels)),#EfficientNet (indice, vitfeat)
        ),
        InitCheckPoint=[
            dict(
               varname='to_patch_embedding.compressai',
               prefix='g_a.',
               checkpoint='https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-5-866ba797.pth.tar',
            ),
            dict(
               varname='convPlus.convBackbone',
               prefix='backbone.',
               checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty-ra-noisystudent_in1k_20221103-756bcbc0.pth',
            ),
            dict(
                varname='transformer',
                prefix='backbone.transformer',
                checkpoint='../../checkpoint/ci2pvit_imagenet/myvar_epoch_63.pth',
            ),
            
        ],
    ),
    neck=dict(type='GapNeck', extCfg=dict(gapsize=gapsize)),# if not WITH_CLS_TOKEN else None,
    head=dict(
        type='LinearClsHead',
        num_classes=CLASS_NUM,
        in_channels=in_channels,
        #loss=dict(type='CrossEntropyLoss', loss_weight=1.0,),
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1,), cal_acc=SHOW_TRAIN_ACCURACY),
    
)
