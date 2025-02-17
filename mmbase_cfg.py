import sys
import os
from mmpretrain.models.backbones import vision_transformer as MMViT
custom_imports = dict(imports='base_model')

sys.path.append('./')
import platform
OS_NAME = platform.system()
print('OS_NAME', OS_NAME)
def IsWin():
    if ('Windows' == OS_NAME):
        return True
    return False
def IsNotLinux():
    if (OS_NAME in ['Windows','Darwin']):
        return True
    return False
IMAGENET_DATASET_DIR_LIST = ['/gm-data/imagenet/','../../dataset/ImageNet_dataset/','/root/autodl-tmp/imagenet/']
IMAGENET_DATASET_DIR = IMAGENET_DATASET_DIR_LIST[0]
for m in IMAGENET_DATASET_DIR_LIST:
    if os.path.isdir(m):
        IMAGENET_DATASET_DIR = m
        break
print('IMAGENET_DATASET_DIR', IMAGENET_DATASET_DIR)

ExtCnnFeat = 1#是否额外添加ConNeXt的特征图
IMG_SIZE = 256#256#256#256#256
PATCH_SIZE = 16
PATCH_DIM = 768
PATCH_LEN = 64
DIM_FEAT = PATCH_DIM#patch 维度
#arch_zoo = {
#        **dict.fromkeys(
#            ['s', 'small'], {
#                'embed_dims': 768,
#                'num_layers': 8,
#                'num_heads': 8,
#                'feedforward_channels': 768 * 3,
#            }),
#        **dict.fromkeys(
#            ['b', 'base'], {
#                'embed_dims': 768,
#                'num_layers': 12,
#                'num_heads': 12,
#                'feedforward_channels': 3072
#            }),
#}
VIT_SUB_TYPE = 'base'#'base'#'base'#'base'#'base'
VIT_DEPTH = MMViT.VisionTransformer.arch_zoo[VIT_SUB_TYPE]['num_layers']

CLASS_NUM = 0
SHOW_TRAIN_ACCURACY = False
MAX_EPOCH = 500

DatasetFlag = 1#flower animal 
BATCH_SIZE_WIN = 32##32

FEATURE_POOL = 'mean'
DIM_HEAD_NUM = 12
LR_RATE = 1e-04
VIT_CNN_CHANNLE = 3




if IsNotLinux():
    IMG_SIZE = 256#256#256#256#256
    #VIT_DEPTH = 6
    BATCH_SIZE_WIN = 10##32
    SHOW_TRAIN_ACCURACY = True

    
BATCH_SIZE = 128#260
if VIT_DEPTH == 12:
    BATCH_SIZE = 128
elif VIT_DEPTH == 8:
    BATCH_SIZE = 256
elif VIT_DEPTH == 6:
    BATCH_SIZE = 500
DatasetPath = ("C:/myai/dataset/flower_photos_dataset", "../../dataset/Animals-10_dataset", "G:/ai/dataset/102flowers_dataset")
cfgDefault = {#win linux
    '--data-path':(DatasetPath[0], DatasetPath[DatasetFlag]),
    '--bs':(BATCH_SIZE_WIN, BATCH_SIZE),
}
def GetCfgByKey(key):
    cfg = cfgDefault.get(key)
    if not cfg:
        return ''
    if IsNotLinux():
        return str(cfg[0])
    return str(cfg[1])
BATCH_SIZE = int(GetCfgByKey('--bs'))
try:
    allclassnum = os.listdir(GetCfgByKey('--data-path')+'/train')
except:
    allclassnum = ['None']
CLASS_NUM = len(allclassnum)
print('allclassnum', CLASS_NUM, allclassnum)


IMG_NORMALIZATION = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    #mean=[0, 0, 0],
    #std=[255, 255, 255],
)

data_preprocessor = dict(
    num_classes=CLASS_NUM,
    # RGB format normalization parameters
    mean=IMG_NORMALIZATION['mean'],
    std=IMG_NORMALIZATION['std'],
    # convert image from BGR to RGB
    #type='ToPIL', 
    #mean=[0, 0, 0],
    #std=[255, 255, 255],
    to_rgb=True
)
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=IMG_SIZE,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    #dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.1),
    dict(type='Solarize', thr=128, prob=0.2),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=int(292/256.0*IMG_SIZE),  # ( 256 / 224 * 256 )
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=IMG_SIZE),
    dict(type='PackInputs'),
]
BATCH_SIZE = int(GetCfgByKey('--bs'))
NUM_WORKER = 5
# dataset settings
dataset_type = 'CustomDataset'
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    dataset=dict(
        type=dataset_type,
        data_root=GetCfgByKey('--data-path'),
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
train_evaluator = dict(type='Accuracy', topk=(1,))

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    dataset=dict(
        type=dataset_type,
        data_root=GetCfgByKey('--data-path'),
        data_prefix='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,))

test_dataloader = val_dataloader
test_evaluator = val_evaluator




# model setting
train_cfg_augments=dict(augments=[
    dict(type='Mixup', alpha=0.8),
    dict(type='CutMix', alpha=1.0)
])
WITH_CLS_TOKEN = False

ViT_Backbone = dict(
    type='VisionTransformer',
    #arch='b',
    arch=dict({
        '_delete_':True,
        'embed_dims': PATCH_DIM,#768,
        'num_layers': VIT_DEPTH,#12,
        'num_heads': MMViT.VisionTransformer.arch_zoo[VIT_SUB_TYPE]['num_heads'] if PATCH_DIM == 768 else PATCH_DIM//64,#base=12,
        'feedforward_channels': MMViT.VisionTransformer.arch_zoo[VIT_SUB_TYPE]['feedforward_channels']#3072
    }),
    out_type='cls_token' if WITH_CLS_TOKEN else 'raw', with_cls_token=WITH_CLS_TOKEN,
    
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    #drop_rate=0.1,
    drop_rate=0, drop_path_rate=0.1, 
    init_cfg=[
        dict(
            type='Kaiming',
            layer='Conv2d',
            mode='fan_in',
            nonlinearity='linear')
    ]
)
model = dict(
    type='ImageClassifier',
    #image_size=256, patch_size=16, dim=768, depth=6, heads=8, mlp_dim = 512, pool = 'mean', dim_head = 64,init_cfg=None)
    backbone=ViT_Backbone,
    neck=dict(type='GapNeck'),# if not WITH_CLS_TOKEN else None,
    #neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='LinearClsHead',
        num_classes=CLASS_NUM,
        in_channels=DIM_FEAT,
        #loss=dict(type='CrossEntropyLoss', loss_weight=1.0,),
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1,), cal_acc=SHOW_TRAIN_ACCURACY),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=train_cfg_augments if not SHOW_TRAIN_ACCURACY else dict(),
)


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=LR_RATE, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=280,
        by_epoch=True,
        begin=20,
        end=MAX_EPOCH,
        eta_min=1.25e-06)
]
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
#optim_wrapper = dict(
#    optimizer=dict(
#        type='AdamW',
#        lr=1e-4 * 4096 / 256,
#        weight_decay=0.3,
#        eps=1e-8,
#        betas=(0.9, 0.95)),
#    paramwise_cfg=dict(
#        norm_decay_mult=0.0,
#        bias_decay_mult=0.0,
#        flat_decay_mult=0.0,
#        #custom_keys={
#        #    '.absolute_pos_embed': dict(decay_mult=0.0),
#        #    '.relative_position_bias_table': dict(decay_mult=0.0)
#        #}
#    ),
#)

# learning policy
#param_scheduler = [
#    # warm up learning rate scheduler
#    dict(
#        type='LinearLR',
#        start_factor=1e-3,
#        by_epoch=True,
#        end=20,
#        # update by iter
#        convert_to_iter_based=True),
#    # main learning rate scheduler
#    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20, T_max=280,end=MAX_EPOCH,)
#]
# runtime settings

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=MAX_EPOCH, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=BATCH_SIZE)
default_hooks = dict( checkpoint=dict(type='CheckpointHook', interval=1),)
VIT_TYPE = 0
VIT_P16_BASE_CHECKPOINT = 'https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth'
CI2PVIT_CHECKPOINT = '../../checkpoint/ci2pvit_imagenet/imagenet_epoch_385.pth'
