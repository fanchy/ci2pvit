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
model.update(
    #image_size=256, patch_size=16, dim=768, depth=6, heads=8, mlp_dim = 512, pool = 'mean', dim_head = 64,init_cfg=None)
    backbone=dict(#_delete_=True,
        type='CI2PVit', img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM_FEAT, depth=VIT_DEPTH, vitBackbone=ViT_Backbone,cai_checkpoint='checkpoint_best_loss.pth.tar'),
    
)
