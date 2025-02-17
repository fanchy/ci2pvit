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
BATCH_SIZE = 128#420
custom_imports = dict(imports='base_model')

VIT_TYPE = 0
CNNPLUS_MERGE_TYPE = 2#1转成featuremap 2 cat 3 add
model.update(
    #image_size=256, patch_size=16, dim=768, depth=6, heads=8, mlp_dim = 512, pool = 'mean', dim_head = 64,init_cfg=None)
    backbone=dict(#_delete_=True,
        type='CI2PVit', img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM_FEAT, depth=VIT_DEPTH, vitBackbone=ViT_Backbone,
        extCfg=dict(
            IMG_NORMALIZATION=IMG_NORMALIZATION,
            cnnPlusMergeIndex=6,
        ),
        vittype=VIT_TYPE,
        ConvPlusCfg = dict(
            plusCNNModelCfg=dict(type='MobileNetV3', arch='large'),
            plusCNNMergeDim=DIM_FEAT if CNNPLUS_MERGE_TYPE!=2 else 960,
            #cnnPlusDetach=True,
        ),
        InitCheckPoint=[
            dict(
               varname='to_patch_embedding.compressai',
               prefix='g_a.',
               checkpoint='https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-5-866ba797.pth.tar',
            ),
            #dict(
            #    varname='transformer',
            #    prefix='backbone.',
            #    checkpoint=VIT_P16_BASE_CHECKPOINT if VIT_TYPE else CI2PVIT_CHECKPOINT,
            #),
            dict(
               varname='convPlus.convBackbone',
               prefix='backbone.',
               checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth',
            ),
            dict(
                varname='transformer',
                prefix='backbone.transformer',
                checkpoint='../../checkpoint/ci2pvit_imagenet/imagenet_epoch_385.pth',
            ),
        ],
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=CLASS_NUM,
        in_channels=DIM_FEAT if CNNPLUS_MERGE_TYPE!=2 else DIM_FEAT+960,
        #loss=dict(type='CrossEntropyLoss', loss_weight=1.0,),
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1,), cal_acc=SHOW_TRAIN_ACCURACY),
    
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
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    dataset=dict(
        type='ImageNet',
        data_root=IMAGENET_DATASET_DIR,
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
default_hooks = dict( checkpoint=dict(type='CheckpointHook', interval=1),)
auto_scale_lr = dict(base_batch_size=BATCH_SIZE*3)