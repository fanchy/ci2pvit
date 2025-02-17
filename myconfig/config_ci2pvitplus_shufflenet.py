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
#if 'arch' in model['backbone']:
#    del model['backbone']['arch']

VIT_TYPE = 0
CNNPLUS_MERGE_TYPE = 2#1转成featuremap 2 cat 3 add
model.update(
    #image_size=256, patch_size=16, dim=768, depth=6, heads=8, mlp_dim = 512, pool = 'mean', dim_head = 64,init_cfg=None)
    backbone=dict(#_delete_=True,
        type='CI2PVit', img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM_FEAT, depth=VIT_DEPTH, vitBackbone=ViT_Backbone, 
        extCfg=dict(
            IMG_NORMALIZATION=IMG_NORMALIZATION,
        ),
        vittype=VIT_TYPE,
        ConvPlusCfg = dict(
            plusCNNModelCfg=dict(type='ShuffleNetV2', widen_factor=1.0),
            plusCNNMergeDim=DIM_FEAT if CNNPLUS_MERGE_TYPE!=2 else 1024,
        ),
        
        InitCheckPoint=[
            dict(
                varname='transformer',
                prefix='backbone.',
                checkpoint=VIT_P16_BASE_CHECKPOINT if VIT_TYPE else CI2PVIT_CHECKPOINT,
            ),
            dict(
               varname='to_patch_embedding.compressai',
               prefix='g_a.',
               checkpoint='https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-5-866ba797.pth.tar',
            ),
            dict(
               varname='convPlus.convBackbone',
               prefix='backbone.',
               checkpoint='https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth',
            ),
        ],
    ),
    neck=dict(type='GapNeck', toFeatMap=CNNPLUS_MERGE_TYPE),
    head=dict(
        type='LinearClsHead',
        num_classes=CLASS_NUM,
        in_channels=DIM_FEAT if CNNPLUS_MERGE_TYPE!=2 else DIM_FEAT+1024,
        #loss=dict(type='CrossEntropyLoss', loss_weight=1.0,),
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1,), cal_acc=SHOW_TRAIN_ACCURACY),
    
)