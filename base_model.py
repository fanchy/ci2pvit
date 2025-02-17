
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import sys
from mmpretrain import get_model
from mmpretrain.registry import MODELS
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.models.utils.inverted_residual import InvertedResidual
import mmobj
from  mmobj import BuildInvertedResidual
import math
import numpy as np


CAI_MODEL_TYPE          = 6#6
COMPRESSED_DIM          = 192
if CAI_MODEL_TYPE in (0,1):
    COMPRESSED_DIM          = 128
LOSS_TYPE = 0#0 mse 1 ms-ssim

def buildModel(modType, cai_checkpoint=None):
    import compressai
    CAI_PRETRAINED = True
    if cai_checkpoint:
        CAI_PRETRAINED = False
    net = None
    metric="mse" if not LOSS_TYPE else 'ms-ssim'
    if modType == 0:#67.212148736 G
        net = compressai.zoo.cheng2020_attn(quality=3, pretrained=CAI_PRETRAINED, metric=metric)
    elif modType == 1:#60.634431488 G
        net = compressai.zoo.cheng2020_anchor(quality=3, pretrained=CAI_PRETRAINED, metric=metric)
    elif modType == 2:#29.464949248 G
        net = compressai.zoo.mbt2018(quality=4, pretrained=CAI_PRETRAINED, metric=metric)
    elif modType == 3:#28.74310656 G 有多余的斑点
        net = compressai.zoo.mbt2018_mean(quality=4, pretrained=CAI_PRETRAINED, metric=metric)
    elif modType == 4:#27.39044352 G 图像边缘有多余斑点
        net = compressai.zoo.bmshj2018_hyperprior(quality=5, pretrained=CAI_PRETRAINED, metric=metric)
    elif modType == 5:#26.738688 G
        net = compressai.zoo.bmshj2018_factorized(quality=5, pretrained=CAI_PRETRAINED, metric=metric)
    elif modType == 6:#26.738688 G
        net = compressai.zoo.bmshj2018_factorized_relu(quality=5, pretrained=CAI_PRETRAINED, metric=metric)
    if cai_checkpoint:
        print("Loading", cai_checkpoint)
        checkpoint = torch.load(cai_checkpoint)#, map_location=device
        net.load_state_dict(checkpoint["state_dict"])
    print('buildModel', __name__, modType, cai_checkpoint)
    return net

class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ConvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, kernel_size=3, NormFlag = 1,**kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel) if NormFlag else nn.Identity()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel) if NormFlag else nn.Identity()
        self.downsample = downsample
        if downsample == True:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)  if NormFlag else nn.Identity())

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #print('convbloc', out.shape, identity.shape)
        out += identity
        out = self.relu(out)

        return out

class PatchReshape(nn.Module):
    def __init__(self):
        super().__init__()
        #self.cnn_reshape = nn.Sequential(
        #    ConvBasicBlock(COMPRESSED_DIM, COMPRESSED_DIM, 1),
        #    ConvBasicBlock(COMPRESSED_DIM, COMPRESSED_DIM*4, 2, True),
        #)
        self.cnn_reshape = nn.Sequential(
            BuildInvertedResidual(COMPRESSED_DIM, COMPRESSED_DIM*4, COMPRESSED_DIM),
            BuildInvertedResidual(COMPRESSED_DIM, COMPRESSED_DIM*6, COMPRESSED_DIM*4, 5, 2),
        )
    def forward(self, x, reshapeFlag = True):
        ret = self.cnn_reshape(x)#b c h w

        if reshapeFlag:
            ret = ret.view(ret.size(0), ret.size(1),ret.size(2)*ret.size(3))
            ret = ret.transpose(1, 2)
        return ret

DUMP_FLAG = 0
def DumpImg(x, ci2p, modeTrainCount):
    global DUMP_FLAG
    if not DUMP_FLAG:
        return
    outputdir = 'output'
    import os
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    #if modeTrainCount < 5:
    #    return
    DUMP_FLAG -= 1
    print('DumpImg x', modeTrainCount, x.shape, x.min(), x.max())
    modCai = buildModel(CAI_MODEL_TYPE).to(device=x.device)
    modCai.g_a = ci2p.compressai
    ret = modCai(x)
    x_hat = ret['x_hat']
    
    if '../simple_i2p' not in sys.path:
        sys.path.append('../simple_i2p')
    import tensor_img
    #y = torch.cat((batch_x[0].cpu().data, fake[0].cpu().data), dim=2).unsqueeze(0)
    #print('xxx', data[0].shape, out[0].shape, y.shape)
    #save_imgs(imgs=y, to_size=(3, y.shape[1], 2 * y.shape[2]), name=fname,)
    IMG_CV_RESIZE = (x.shape[-2],x.shape[-1])
    from torchvision import transforms
    TFCFG = transforms.Compose([
        transforms.Resize(IMG_CV_RESIZE[0]),
        transforms.RandomCrop(IMG_CV_RESIZE),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],)
    ])
    for i in range(x.size(0)):
        fname = '%s/traindump%d_%d.png'%(outputdir, DUMP_FLAG, i)
        #tensor_img.cv2_show_tensor(y[0], 'ingore', True, str(fname))
        tensor_img.ShowTensor2Img(x[i].cpu(), x_hat[i].cpu(), TFCFG, True, str(fname), False, [], [0,1])
    return
def BuildCnnModelPlus(pretrained=False):#mobilenet-v3-small_3rdparty_in1k mobilenet-v3-large_3rdparty_in1k
    cnnExtModel = get_model('mobilenet-v3-large_3rdparty_in1k', pretrained=pretrained)#576 8 8
    return cnnExtModel
class ConvPlus(nn.Module):
    def __init__(self, ConvPlusCfg:dict = None):
        super().__init__()
        if not ConvPlusCfg:
            ConvPlusCfg = {}
        self.extCNNPretrain = ConvPlusCfg.get('extCNNPretrain', False)
        self.cnnPlusDetach = ConvPlusCfg.get('cnnPlusDetach', False)
        #cnnExtModel = BuildCnnModelPlus(extCNNPretrain)
        #
        #inc = 320
        #if cnnExtModel.head.__class__.__name__ == 'StackedLinearClsHead':
        #    inc = cnnExtModel.head.layers[0].fc.in_features
        #else:
        #    inc = cnnExtModel.head.fc.in_features
        #self.convBackbone = cnnExtModel.backbone
        plusCNNModelCfg = ConvPlusCfg.get('plusCNNModelCfg', None)
        #plusCNNFeatDim = ConvPlusCfg.get('plusCNNFeatDim', 960)

        if not plusCNNModelCfg:
            plusCNNModelCfg = dict(type='MobileNetV3', arch='large')
        self.convBackbone = MODELS.build(plusCNNModelCfg)
        a = torch.randn(1, 3, 256, 256)
        tmpfeat = self.convBackbone(a)
        plusCNNFeatDim = tmpfeat[-1].size(1)
        
        plusCNNMergeDim = ConvPlusCfg.get('plusCNNMergeDim', COMPRESSED_DIM*4)
        plusCNNMergeStep = ConvPlusCfg.get('plusCNNMergeStep')
        print('ConvPlus', plusCNNModelCfg, plusCNNFeatDim, self.cnnPlusDetach, len(tmpfeat), plusCNNMergeStep)
        self.plusCNNMergeStep = plusCNNMergeStep
        if plusCNNMergeStep:
            self.plusCNNMergeStepMod = nn.Sequential()
            for k in plusCNNMergeStep:
                indim = tmpfeat[k[0]].size(1)
                mod = nn.Sequential(
                    nn.Conv2d(indim+k[1], k[2] , kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(k[2]),
                    nn.ReLU(inplace=True),
                )
                self.plusCNNMergeStepMod.append(mod)
        else:
            self.convBackboneAfter = nn.Sequential(#nn.Conv2d(plusCNNFeatDim, COMPRESSED_DIM*4, kernel_size=1, stride=1, bias=False)
                nn.Conv2d(plusCNNFeatDim, COMPRESSED_DIM*4 , kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(COMPRESSED_DIM*4),
                nn.ReLU(inplace=True),
            )
            self.featFuse = nn.Sequential(
                nn.Conv2d(plusCNNFeatDim+COMPRESSED_DIM*4, plusCNNFeatDim+COMPRESSED_DIM*4 , kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(plusCNNMergeDim+COMPRESSED_DIM*4),
                nn.ReLU(inplace=True),
            )
        
        #self._freeze_stages()
    def fuseConvPlus(self, xvit, xconv):
        dim = xvit.size(2)
        xvit = xvit.transpose(1,2).view(xvit.size(0), dim, 8, 8)
        xfeat = torch.cat([xvit, xconv], dim=1)
        #print('fuseConvPlus', xfeat.shape)
        featFuse = self.featFuse(xfeat.detach() if self.cnnPlusDetach else xfeat)
        
        ret = featFuse.view(featFuse.size(0), featFuse.size(1),featFuse.size(2)*featFuse.size(3))
        ret = ret.transpose(1, 2)
        return ret
    def forward_cnn(self, xsrc):
        return self.convBackbone(xsrc)
    def fuseConvStep(self, xvit, xconvAll, step):
        if len(xvit.shape) == 3:
            dim = xvit.size(2)
            hw = int(math.sqrt(xvit.size(1)))
            xvit = xvit.transpose(1,2).view(xvit.size(0), dim, hw, hw)
        indice = self.plusCNNMergeStep[step][0]
        xconv = xconvAll[indice]
        xfeat = torch.cat([xvit, xconv], dim=1)
        featFuse = self.plusCNNMergeStepMod[step]
        featFuse = featFuse(xfeat)
        #print('fuseConvStep', step, xvit.shape, xfeat.shape, featFuse.shape)
        
        ret = featFuse.view(featFuse.size(0), featFuse.size(1),featFuse.size(2)*featFuse.size(3))
        ret = ret.transpose(1, 2)
        return ret
    def forward(self, xsrc):
        xfeat = self.convBackbone(xsrc)[-1]
        ret = self.convBackboneAfter(xfeat)#b 768 8x8
        
        ret = ret.view(ret.size(0), ret.size(1),ret.size(2)*ret.size(3))
        ret = ret.transpose(1, 2)

        #xfeat = xfeat.view(xfeat.size(0), xfeat.size(1),xfeat.size(2)*xfeat.size(3))
        #xfeat = xfeat.transpose(1, 2)
        return ret, xfeat
    #def train(self, mode=True):
    #    super().train(mode)
    #    self._freeze_stages()
    #    
    #    #print('train CAIPatchEmbed', mode)
    #def _freeze_stages(self):
    #    if not self.extCNNPretrain:
    #        return
    #    self.convBackbone.eval()
    #    for param in self.convBackbone.parameters():
    #        param.requires_grad = False
        
#ci2p 0 torch.Size([10, 128, 128, 128])
#ci2p 1 torch.Size([10, 128, 128, 128])
#ci2p 2 torch.Size([10, 128, 64, 64])
#ci2p 3 torch.Size([10, 128, 64, 64])
#ci2p 4 torch.Size([10, 128, 32, 32])
#ci2p 5 torch.Size([10, 128, 32, 32])
#ci2p 6 torch.Size([10, 192, 16, 16])
#7 torch.Size([10, 192, 16, 16])    
class CI2P(nn.Module):
    def __init__(self, extCfg:dict = None, cai_checkpoint=None):
        super().__init__()
        if not extCfg:
            extCfg = {}
        ci2pDownSample = extCfg.get('ci2pDownSample', True)
        self.patchFlatten = extCfg.get('patchFlatten', 0)
        dimOut = extCfg.get('ci2pDimout', COMPRESSED_DIM*4)
        if dimOut != COMPRESSED_DIM*4:
            ci2pDownSample = False
        conv_kernelsize = extCfg.get('conv_kernelsize', 5)
        self.compressai = buildModel(CAI_MODEL_TYPE, cai_checkpoint).g_a
        self.patchReshape = nn.Sequential(
            #ConvBasicBlock(COMPRESSED_DIM, COMPRESSED_DIM, 1),
            #ConvBasicBlock(COMPRESSED_DIM, COMPRESSED_DIM*4, 2, True),
            BuildInvertedResidual(COMPRESSED_DIM, COMPRESSED_DIM*4, COMPRESSED_DIM),
            BuildInvertedResidual(COMPRESSED_DIM, COMPRESSED_DIM*6, dimOut, conv_kernelsize, 2) if ci2pDownSample else BuildInvertedResidual(COMPRESSED_DIM, COMPRESSED_DIM*4, dimOut),
        )
        if extCfg.get('ci2pPatchReshapeClose', False):
            self.patchReshape = nn.Identity()
            print('ci2pPatchReshapeClose', extCfg.get('ci2pPatchReshapeClose', False))
  
        self.modeTrainCount = 0
        self._freeze_stages()
        
    def forward(self, xsrc, out_indices=-1):
        x = xsrc
        
        DumpImg(x, self, self.modeTrainCount)
        if out_indices == -1:
            xq1 = self.compressai(x)
            ret = self.patchReshape(xq1)
            if self.patchFlatten != 1:#b c h w
                ret = ret.view(ret.size(0), ret.size(1),ret.size(2)*ret.size(3))
                ret = ret.transpose(1, 2)
            return ret
        retList = []
        xq1 = x
        for mod in self.compressai:
            xq1 = mod(xq1)
            retList.append(xq1)
        ret = self.patchReshape(xq1)
        retList.append(ret)
        return retList
    def train(self, mode=True):
        self.modeTrainCount += 1
        super().train(mode)
        self._freeze_stages()
        
        #print('train CAIPatchEmbed', mode)
    def _freeze_stages(self):
        self.compressai.eval()
        for param in self.compressai.parameters():
            param.requires_grad = False


@MODELS.register_module()
class GapNeck(nn.Module):
    def __init__(self, toFeatMap=0, extCfg = None):
        super().__init__()
        self.toFeatMap = toFeatMap
        self.gap = None
        if not extCfg:
            extCfg = {}
        gapsize = extCfg.get('gapsize', None)
        if gapsize:
            self.gap = nn.AdaptiveAvgPool2d((gapsize, gapsize))
            self.toFeatMap = 1
        
    def forward(self, inputs):
        #print('GapNeck', inputs[0].shape)
        if self.toFeatMap == 1:#再转回featureamap方便做上采样
            ret = []
            for k in inputs:
                if len(k.shape) == 3:
                    k = k.transpose(1, 2)
                    hw = int(math.sqrt(k.shape[-1]))
                    #print('neck1', k.shape, hw)
                    k2 = k.reshape(k.size(0), k.size(1), hw, hw)
                    #print('neck', k.shape, k2.shape)
                    ret.append(k2)
                else:
                    ret.append(k)
            if self.gap is not None:
                feat = ret[-1]
                #print('feat', feat.shape)
                tmpmean= self.gap(feat)
                tmpmean = tmpmean.reshape(tmpmean.size(0), -1)
                ret=(tmpmean,)
                #print('tmpmean', tmpmean.shape)
                #raise 1
            outs = tuple(ret)
            return outs
        elif self.toFeatMap == 2:#特征连接
            outs = tuple([x.mean(dim = 1) for x in inputs])
            ret = torch.cat(outs, dim=1)
            #print('xx', ret.shape)
            outs = (ret,)
            return outs
        elif self.toFeatMap == 3:#特征相加再取均值
            xmerge = inputs[0]
            for i in range(1, len(inputs)):
                xmerge = xmerge + inputs[i]
            ret = xmerge.mean(dim = 1)
            #print('xx', ret.shape)
            outs = (ret,)
            return outs
        elif self.toFeatMap == 4:#下采样再取均值 输出结果 dim=192x2Hx2W->768xHxW
            tmplist = []
            for x in inputs:
                k = x.transpose(1, 2)
                hw = int(math.sqrt(k.shape[-1]))
                #print('k', k.shape)
                newx = rearrange(k.view(k.size(0), k.size(1), hw, hw), 'b d (m h) (n w) -> b (d m n) h w', m=2, n=2)
                newx = newx.view(newx.size(0), newx.size(1), -1).transpose(1, 2)
                tmplist.append(newx)
            outs = tuple([x.mean(dim = 1) for x in tmplist])
            return outs
        elif self.toFeatMap == 5:#再转回featureamap方便做上采样
            ret = [inputs[-2]]
            if 1:
                k = inputs[-1]
                if len(k.shape) == 3:
                    k = k.transpose(1, 2)
                    hw = int(math.sqrt(k.shape[-1]))
                    #print('neck1', k.shape, hw)
                    k2 = k.reshape(k.size(0), k.size(1), hw, hw)
                    #print('neck', k.shape, k2.shape)
                    ret.append(k2)
                else:
                    ret.append(k)
            outs = tuple(ret)
            return outs
        if isinstance(inputs, tuple):
            if len(inputs[-1].shape) == 2:#class token
                return inputs
            outs = (inputs[-1].mean(dim = 1),)#默认返回最后一个
        elif isinstance(inputs, torch.Tensor):
            outs = inputs.mean(dim = 1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor:'+str(type(inputs)))
        return outs

@MODELS.register_module()
class AttentionMergeNeck(nn.Module):
    def __init__(self, in_channels = 768, toFeatMap=0):
        super().__init__()
        self.fccor = nn.Linear(in_channels, in_channels)
        self.attend = nn.Softmax(dim = -1)
        dim_head = 256
        self.q = nn.Linear(in_channels, dim_head)
        self.k = nn.Linear(in_channels, dim_head)
        self.scale = dim_head ** -0.5
    def forward2(self, inputs):#inputs[0]=cnn inputs[1]=vit
        featall = torch.cat(inputs, dim=1)
        q = self.q(featall)
        k = self.k(featall)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, featall)

        ret = out.mean(dim = 1)
        
        #print('featall', featall.shape, attn.shape, out.shape, ret.shape)
        #raise 1
        return (ret,)
    def forward(self, inputs):#inputs[0]=cnn inputs[1]=vit
        featall = torch.cat([x.mean(dim = 1).unsqueeze(1) for x in inputs], dim=1)
        qvfeat = self.fccor(featall).transpose(1, 2)
        attn = self.attend(qvfeat)
        ret = featall[:,0,:]*attn[:,:,0] + featall[:,1,:]*attn[:,:,1]
        
        #print('featall', featall.shape, qvfeat.shape, attn.shape, ret.shape)
        return (ret,)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def CalVolo(trans, x, ci2pvitObj):
    #x = trans.forward_embeddings(x)
    # B,C,H,W-> B,H,W,C
    x = x.permute(0, 2, 3, 1)
    #print('CalVolo 1', x.shape)
    # step2: tokens learning in the two stages
    if 1:
        x = trans.forward_tokens(x)
    else:
        for idx, block in enumerate(trans.network):
            if idx == 2:  # add positional encoding after outlooker blocks
                x = x + trans.pos_embed
                x = trans.pos_drop(x)
            x = block(x)
        B, H, W, C = x.shape
        #print('CalVolo 2', idx, x.shape)
        x = x.reshape(B, -1, C)
    
    #print('CalVolo 3', x.shape)

    # step3: post network, apply class attention or not
    if trans.post_network is not None:
        x = trans.forward_cls(x)
    x = trans.norm(x)
    x_cls = x[:, 0]
    #print('x_cls', x.shape)
    return (x_cls,)
def CalMyVitVar2(trans, img, ci2pvitObj, mix_token = True, beta = 1.0):
    xlist = ci2pvitObj.to_patch_embedding(img)
    x = xlist[-1]
    if mix_token and ci2pvitObj.training:
        lam = np.random.beta(beta, beta)
        d = x.size(2)
        hw = int(math.sqrt(x.shape[1]))
        patch_h, patch_w = hw, hw#x.shape[2],x.shape[3]
        x = x.transpose(1, 2).view(x.size(0), d, hw, hw)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        temp_x = x.clone()
        temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
        x = temp_x
        x = x.view(x.size(0), d, hw*hw).transpose(1, 2)
        #print('CalMyVitVar', x.shape, bbx1, bby1, bbx2, bby2)
    else:
        bbx1, bby1, bbx2, bby2 = 0,0,0,0
    if 1:
        x = trans(x)[-1]
    else:
        B = x.shape[0]
        cls_tokens = ci2pvitObj.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + ci2pvitObj.pos_embed

        x = trans.drop_after_pos(x)
        x = trans.pre_norm(x)
        outs = []
        #print('CalTransformer', x.shape)
        for i, layer in enumerate(trans.layers):
            x = layer(x)

            if i == len(trans.layers) - 1 and trans.final_norm:
                x = trans.ln1(x)

            if i in trans.out_indices:
                retfeat = trans._format_output(x, trans.patch_resolution)
                #if len(retfeat.shape) == 2:
                #    retfeat = retfeat.unsqueeze(1)
                outs.append(retfeat)
        x = outs[-1]
    x_cls = ci2pvitObj.head(x[:,0])
    return_dense = True
    if return_dense:
        x_aux = ci2pvitObj.aux_head(x[:,1:])
        if not ci2pvitObj.training:
            return x_cls+0.5*x_aux.max(1)[0]
        if mix_token and ci2pvitObj.training:
            x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1])
            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1])
        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
    return x_cls
def CalMyVitVar(trans, img, ci2pvitObj, mix_token = True, beta = 1.0):
    xlist = ci2pvitObj.to_patch_embedding(img, 0)
    x = xlist[-3]

    B = x.shape[0]

    #if len(x.shape) == 4:#b c h w
    #    x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
    #x = trans.drop_after_pos(x)
    #x = trans.pre_norm(x)
    outs = []
    #print('CalTransformer', x.shape)

    for i, layer in enumerate(trans.layers):
        #print('myvit trace', i, x.shape)
        if i == 0:
            x = trans.dimProj[0](x)#128 32x32-> 128 32x32
            x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
            x = trans.drop_after_pos(x)
            x = trans.pre_norm(x)
        elif i == 2:
            hw = int(math.sqrt(x.shape[1]))
            dim = x.size(2)
            xconv = x.transpose(1, 2).view(x.size(0), dim, hw, hw)
            xnew = trans.dimProj[1](xconv)#128 32x32-> 192 16x16
            xraw = trans.dimProj[2](xlist[-2])
            xmerge = xraw + xnew
            x = trans.dimMerge[0](xmerge)
            x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        elif i == 6:
            hw = int(math.sqrt(x.shape[1]))
            dim = x.size(2)
            xconv = x.transpose(1, 2).view(x.size(0), dim, hw, hw)
            x = trans.dimProj[3](xconv)#192 16x16-> 768 8x8
            x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        #print('myvit', i, x.shape)
        x = layer(x)

        if i == len(trans.layers) - 1 and trans.final_norm:
            x = trans.ln1(x)

        if i in trans.out_indices:
            retfeat = trans._format_output(x, trans.patch_resolution)
            #if len(retfeat.shape) == 2:
            #    retfeat = retfeat.unsqueeze(1)
            outs.append(retfeat)
    return tuple(outs)
def MyVarImpl(trans, img, ci2pvitObj):
    ret = trans.forward(img, ci2pvitObj)
    if isinstance(ret, tuple):
        return ret
    return (ret,)
def CalTransformer(trans, x, ci2pvitObj, imgraw):
    if ci2pvitObj.vittype == 2:
        return CalVolo(trans, x, ci2pvitObj)
    elif ci2pvitObj.vittype == 3:
        return trans(x, imgraw, ci2pvitObj)

    if trans.pos_embed is not None:
        trans.patch_embed = nn.Identity()
        trans.pos_embed = None
        trans.drop_after_pos = nn.Identity()
    outs = []
    B = x.size(0)
    #patch_resolution = None
    #x2 = torch.rand(1, 3, 224, 224).to(x.device)
    #x2, patch_resolution = trans.patch_embed(x2)
    #print('x2', x2.shape, patch_resolution)

    if ci2pvitObj.pos_embedding is not None:
        x = x + ci2pvitObj.pos_embedding.to(x.device, dtype=x.dtype)
        
    if trans.cls_token is not None:
        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = trans.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

 
    x = trans.drop_after_pos(x)
    x = trans.pre_norm(x)

    #print('CalTransformer', x.shape)
    xConvPlus = None
    for i, layer in enumerate(trans.layers):
        if ci2pvitObj.convPlus is not None and i == ci2pvitObj.cnnPlusMergeIndex:
            xConvPlus = ci2pvitObj.convPlus(imgraw)
            x = x + xConvPlus[0]
            #print('cnnPlusMergeIndex', ci2pvitObj.cnnPlusMergeIndex, xConvPlus[0].shape, xConvPlus[1].shape)
        x = layer(x)

        if i == len(trans.layers) - 1 and trans.final_norm:
            x = trans.ln1(x)

        if i in trans.out_indices:
            retfeat = trans._format_output(x, trans.patch_resolution)
            #if len(retfeat.shape) == 2:
            #    retfeat = retfeat.unsqueeze(1)
            outs.append(retfeat)
    #print('trans.cls_token', trans.cls_token.shape, outs[-1].shape, trans.patch_resolution)
    #raise 1
    if xConvPlus is not None:
        x = ci2pvitObj.convPlus.fuseConvPlus(outs[-1], xConvPlus[1])
        #print('xconvplus fuse', x.shape)
        return (x,)
    return tuple(outs)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

@MODELS.register_module()
class CI2PVit(BaseBackbone):#vittype 0自己构建 1使用现有的
    def __init__(self, img_size=256, patch_size=16, dim=768, depth=12, UsePosEmbed=0, vitBackbone=None, ConvPlusCfg=None, 
                 vittype = 0,
                 InitCheckPoint:dict=None,
                 extCfg:dict = None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.vittype = vittype
        self._enable_normalize = False
        if not extCfg:
            extCfg = {}
        self.withRawData = extCfg.get('withRawData', 0)
        self.mix_token = extCfg.get('mix_token', 0)
        if self.mix_token:
            num_classes = extCfg.get('num_classes', 1000)
            self.head = nn.Linear(dim, num_classes)
            self.aux_head=nn.Linear(dim, num_classes)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size//2)**2 + 1, dim))

        self.cnnPlusMergeIndex = extCfg.get('cnnPlusMergeIndex', 0)
        self.freezeViT = extCfg.get('freezeViT', False)
        imgNormalization = extCfg.get('IMG_NORMALIZATION')
        if imgNormalization:
            self._enable_normalize = True
            mean = imgNormalization['mean']
            std = imgNormalization['std']
            self.register_buffer('mean',
                                    torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                    torch.tensor(std).view(-1, 1, 1), False)
        self.InitCheckPoint = InitCheckPoint if InitCheckPoint else []

        self.to_patch_embedding = CI2P(extCfg)
        self.convPlus = None
        if ConvPlusCfg:
            self.convPlus = ConvPlus(ConvPlusCfg)

        if UsePosEmbed:
            self.pos_embedding = posemb_sincos_2d(
                h = img_size // patch_size//2,
                w = img_size // patch_size//2,
                dim = dim,
            ) 
        else:
            self.pos_embedding = None

        if vittype == 0:
            print('vitBackbone', vitBackbone)
            trans = MODELS.build(vitBackbone)
            if extCfg.get('delUselessParam', True):
                trans.patch_embed = nn.Identity()
                trans.pos_embed = None
                trans.drop_after_pos = nn.Identity()
            self.transformer = trans
        elif vittype == 1:
            model = get_model("vit-base-p16_32xb128-mae_in1k", head=None, neck=None, pretrained=False)
            trans = model.backbone
            #trans = get_model("vit-base-p16_32xb128-mae_in1k", head=None, neck=None, backbone=dict(arch=archcfg,img_size=img_size, out_type='raw', with_cls_token=False)).backbone
            print('trans vitcheckpoint', trans, init_cfg)
            #trans.patch_embed = nn.Identity()
            #trans.pos_embed = None
            #trans.drop_after_pos = nn.Identity()
            self.transformer = trans
            #self.model = model
        elif vittype == 2:
            #model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=False)
            from volo.models import volo
            model = volo.volo_d1()
            print('model', model)
            self.transformer = model
            #raise 1
        elif vittype == 3 or vittype == 4:
            from myvision_transformer import MyVisionTransformer
            trans = MyVisionTransformer(drop_path_rate=vitBackbone.get('drop_path_rate', 0.0), drop_rate=vitBackbone.get('drop_rate', 0.0), extCfg=extCfg)
            if extCfg.get('delUselessParam', True):
                trans.patch_embed = nn.Identity()
                trans.pos_embed = None
                trans.drop_after_pos = nn.Identity()
            self.transformer = trans
            print('self.transformer', self.transformer)
        elif vittype == 10:
            dimEmbed = extCfg.get('ci2pDimout', COMPRESSED_DIM)
            import myceit
            trans = myceit.ceit_base_patch16_256(dimEmbed=dimEmbed)

            self.transformer = trans
            print('self.transformer myceit', self.transformer, dimEmbed)

    
    def forward(self, imgraw):
        #ret = self.transformer(img)
        #print('ret', ret.shape)
        #raise 1
        #return (ret,)
        img = imgraw
        if self._enable_normalize:
            inputs = (img * self.std + self.mean)/255
            #x = (xsrc*self.std +self.mean)/255
            #print('DumpImg xsrc', img.shape, self.mean, self.std, img.min(), img.max(), inputs.min(), inputs.max())
            img = inputs
        if self.vittype == 4:
            return CalMyVitVar(self.transformer, img, self)
        if self.vittype >= 10:
            return MyVarImpl(self.transformer, img, self)
        
        out_indices =  0 if self.withRawData == 2 else -1
        xci2p = self.to_patch_embedding(img, out_indices)
        if out_indices == -1:
            x = xci2p
        else:
            x = xci2p[-1]
            x = x.view(x.size(0), x.size(1),x.size(2)*x.size(3))
            x = x.transpose(1, 2)
            
        x = CalTransformer(self.transformer, x, self, imgraw)#self.transformer(x)

        if self.withRawData == 1:
            return (imgraw, x[-1])
        elif self.withRawData == 2:
            xci2p.append(x[-1])
            return tuple(xci2p)
        return x
    def init_weights(self):
        super().init_weights()
        for m in self.InitCheckPoint:
            mmobj.InitCheckpoint(self, m)
    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        
    def _freeze_stages(self):
        if not self.freezeViT:
            return
        print('train self.freezeViT', self.freezeViT)
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
#

@MODELS.register_module()
class DistilledCI2PViT(CI2PVit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = 768
        self.num_classes = 1000
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = 64#self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        #trunc_normal_(self.dist_token, std=.02)
        #trunc_normal_(self.pos_embed, std=.02)
        #self.head_dist.apply(self._init_weights)

    def forward_features(self, img):
        if self._enable_normalize:
            inputs = (img * self.std + self.mean)/255
            #x = (xsrc*self.std +self.mean)/255
            #print('DumpImg xsrc', img.shape, self.mean, self.std, img.min(), img.max(), inputs.min(), inputs.max())
            img = inputs
        x = self.to_patch_embedding(img)
        B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((dist_token, x), dim=1)

        x = x + self.pos_embed
        
        outs = []
        trans = self.transformer
        x = trans.drop_after_pos(x)
        x = trans.pre_norm(x)

        #print('CalTransformer', x.shape)
        for i, layer in enumerate(trans.layers):
            x = layer(x)

            if i == len(trans.layers) - 1 and trans.final_norm:
                x = trans.ln1(x)

            if i in trans.out_indices:
                retfeat = trans._format_output(x, trans.patch_resolution)
                #if len(retfeat.shape) == 2:
                #    retfeat = retfeat.unsqueeze(1)
                outs.append(retfeat)
        #print('trans.cls_token', trans.cls_token.shape, outs[-1].shape, trans.patch_resolution)
        #raise 1
        
        x = outs[-1]
        xfeat = x[:,1:,:]
        x_cls = xfeat.mean(dim = 1)
        x_dist = x[:, 0]
        #print('xfeat', xfeat.shape, x_cls.shape, x_dist.shape)
        return x_cls, x_dist, xfeat

    def forward(self, x):
        x, x_dist,_ = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

@MODELS.register_module()
class DistilledCI2PViTMyVar(CI2PVit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = 768//4
        self.num_classes = 1000
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = 16*16#self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        #trunc_normal_(self.dist_token, std=.02)
        #trunc_normal_(self.pos_embed, std=.02)
        #self.head_dist.apply(self._init_weights)

    def forward_features(self, img):
        if self._enable_normalize:
            inputs = (img * self.std + self.mean)/255
            #x = (xsrc*self.std +self.mean)/255
            #print('DumpImg xsrc', img.shape, self.mean, self.std, img.min(), img.max(), inputs.min(), inputs.max())
            img = inputs
        x = self.to_patch_embedding(img)
        B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((dist_token, x), dim=1)

        x = x + self.pos_embed

        outs = self.transformer(x)
        
        print('trans.ret', outs[-1].shape)

        x = outs[-1]
        xfeat = x[:,1:,:]
        x_cls = xfeat.mean(dim = 1)
        x_dist = x[:, 0]
        #print('xfeat', xfeat.shape, x_cls.shape, x_dist.shape)
        return x_cls, x_dist, xfeat

    def forward(self, x):
        x, x_dist,_ = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

@MODELS.register_module()
class MobileNetWrap(BaseBackbone):
    def __init__(self, img_size=256, patch_size=16, dim=768, depth=6, ExtCnnFeat=0, UsePosEmbed=0, arch=None, vitBackbone=None, init_cfg=None):
        super().__init__(init_cfg)
        plusCNNModelCfg = dict(type='MobileNetV3', arch='large')
        self.net = MODELS.build(plusCNNModelCfg)

    def forward(self, img):
        x = self.net(img)
        return x

print('load model', __name__)