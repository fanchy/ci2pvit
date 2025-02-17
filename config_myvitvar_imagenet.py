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
CLASS_NUM = 1000
BATCH_SIZE = 160#160
custom_imports = dict(imports='base_model')
model.update(
    #image_size=256, patch_size=16, dim=768, depth=6, heads=8, mlp_dim = 512, pool = 'mean', dim_head = 64,init_cfg=None)
    backbone=dict(#_delete_=True,
        type='CI2PVit', img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM_FEAT, depth=VIT_DEPTH, vitBackbone=ViT_Backbone,
        extCfg=dict(
            IMG_NORMALIZATION=IMG_NORMALIZATION,
            ci2pDownSample=False, 
            ci2pDimout=192,
        ),
        #init_cfg=dict(
        #    type='Pretrained',
        #    prefix='backbone.',
        #    checkpoint='../../checkpoint/ci2pvit_imagenet/imagenet_epoch_385.pth',
        #),
        vittype=3,
        InitCheckPoint=[
            dict(
               varname='to_patch_embedding.compressai',
               prefix='g_a.',
               checkpoint='https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-5-866ba797.pth.tar',
            ),
            #dict(
            #    varname='transformer',
            #    prefix='backbone.',
            #    checkpoint='https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth',
            #),
            #dict(#volo
            #    varname='transformer',
            #    #prefix='backbone.',
            #    checkpoint='../../checkpoint/ci2pvit_imagenet/d1_224_84.2.pth.tar',
            #),
            
        ],
    ),
    neck=dict(type='GapNeck', ),# if not WITH_CLS_TOKEN else None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=CLASS_NUM,
        in_channels=DIM_FEAT,
        #loss=dict(type='CrossEntropyLoss', loss_weight=1.0,),
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1,), cal_acc=SHOW_TRAIN_ACCURACY)
    
)
data_preprocessor.update(
    num_classes=CLASS_NUM,
)

train_dataloader.update(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    dataset=dict(
        type='ImageNet',
        data_root=IMAGENET_DATASET_DIR,
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    dataset=dict(
        type='ImageNet',
        data_root=IMAGENET_DATASET_DIR,
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
default_hooks = dict( checkpoint=dict(type='CheckpointHook', interval=1),)
auto_scale_lr = dict(base_batch_size=BATCH_SIZE*3)