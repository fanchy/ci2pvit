import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import sys
from mmpretrain import get_model
from mmpretrain.registry import MODELS
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.models.utils.inverted_residual import InvertedResidual

import copy
from mmpretrain.models.heads import ClsHead
from mmpretrain.structures import DataSample
from typing import List, Optional, Tuple, Union, Sequence
from mmpretrain.evaluation.metrics import Accuracy
from mmengine.evaluator import BaseMetric
import numpy as np
from mmpretrain.evaluation.metrics.single_label import to_tensor

from mmpretrain.registry import METRICS
from mmengine.model.weight_init import print_log,WEIGHT_INITIALIZERS,build_from_cfg,update_init_info,initialize

def BuildNewTarget(targetsrc: torch.Tensor, defaultToType = None):#torch.float32
    target = targetsrc.view(-1)
    targetExt = target.expand(target.size(0), -1)
    targetCorr = (targetExt== target.view(target.size(0), 1))
    if defaultToType is not None:
        targetCorr = targetCorr.to(defaultToType)
    else:
        targetCorr = targetCorr.to(target.dtype)
    targetCorr = targetCorr.view(-1)
    #print(__name__, 'BuildNewResultAndTarget', cls_score.shape, target.shape, target, targetNew.dtype, targetNew)
    return targetCorr
def BuildNewClsScore(cls_score: torch.Tensor):
    cls_score1 = cls_score.view(-1, 1)
    pred_scores = torch.cat([1-cls_score1, cls_score1], dim=1)
    return pred_scores


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
class AttentionBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
@MODELS.register_module()
class CorrelationClsHead(ClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.proj = FeedForward(self.in_channels, self.in_channels)#nn.Linear(self.in_channels, self.in_channels)#

        #self.attend = nn.Softmax(dim = -1)
        self.register_buffer('classStandard', torch.randn(num_classes, self.in_channels), False)
        self.trainFeatHistory = [[] for _ in range(num_classes)]
        self.maxHistoryNum = 500

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        return feats[-1]
    def cal_class_score(self, pre_logits):
        q = self.proj(pre_logits)
        k = self.proj(self.classStandard)
        scale = pre_logits.shape[-1] ** -0.5
        if 0:
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            cls_score = dots#self.attend(dots)
        else:
            cls_score = torch.cosine_similarity(q.unsqueeze(1), k.unsqueeze(0), dim=-1)
        #print('cls_score', cls_score.shape)
        #raise 1

        return cls_score
    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.proj is None:
            self.proj = FeedForward(self.in_channels, self.in_channels)
            self.proj = self.proj.to(feats[-1].device)
        pre_logits = self.pre_logits(feats)
        cls_score = self.cal_class_score(pre_logits)
        #print(__name__, 'cls_score', cls_score.shape)
        return cls_score
    def update_featStandard(self, pre_logits: Tuple[torch.Tensor], data_samples: List[DataSample]):
        
        allfeat = pre_logits[-1].detach()
        #print('_get_loss', allfeat.shape)
        allLabels = {}
        lableList = None
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            lableList = [i.gt_score for i in data_samples]
        else:
            lableList = [i.gt_label for i in data_samples]
        for i in range(len(lableList)):
            label = int(lableList[i])
            allLabels[label] = True
            classList = self.trainFeatHistory[label]
            classList.append(allfeat[i,:])
            if len(classList) > self.maxHistoryNum:
                classList.pop(0)
            #print('_get_loss', i, label, len(classList))
        #更新缓存
        for label, _ in allLabels.items():
            classList = self.trainFeatHistory[label]
            tmp = torch.stack(classList, dim=0)
            self.classStandard[label] = tmp.mean(dim=0)
            #print('tmp', label, tmp.shape, self.classStandard[label].shape)

        #print(__name__, '_get_loss', cls_score.shape, lableList)
        return
    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],**kwargs) -> dict:
        self.update_featStandard(feats, data_samples)
        cls_score = self(feats)
        cls_score2= self.fc(feats[-1].detach())

        losses = self._get_loss(cls_score, data_samples, cls_score2, **kwargs)
        return losses
    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], cls_score2=None, **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        if cls_score2 is not None:
            newScore = torch.cat([cls_score, cls_score2], dim=0)
            newTarget = torch.cat([target, target], dim=0)
            #print('cls_score2', cls_score.shape, cls_score2.shape, newScore.shape, newTarget.shape)
            loss = self.loss_module(
                newScore, newTarget, avg_factor=newScore.size(0), **kwargs)
            losses['loss'] = loss
            cls_score = cls_score2
        else:
            loss = self.loss_module(
                cls_score, target, avg_factor=cls_score.size(0), **kwargs)
            losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses
    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        cls_score= self.fc(feats[-1])
        #cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions
@METRICS.register_module()
class CorrelationAccuracy(BaseMetric):
    default_prefix: Optional[str] = 'accuracy'

    def __init__(self,
                 topk: Union[int, Sequence[int]] = (1, ),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(topk, int):
            self.topk = (topk, )
        else:
            self.topk = tuple(topk)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs, )
        else:
            self.thrs = tuple(thrs)

    def process(self, data_batch, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                result['pred_label'] = data_sample['pred_label'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        metrics = {}

        # concat
        target = torch.cat([res['gt_label'] for res in results])

        
        if 'pred_score' in results[0]:
            #print(__name__, 'compute_metrics', len(results), target.shape, results[0]['gt_label'].shape, results[0]['pred_score'].shape)

            pred = torch.stack([res['pred_score'] for res in results])
            #print(__name__, 'compute_metrics', target.shape, pred.shape, results[0]['gt_label'].shape, results[0]['pred_score'].shape)

            try:
                acc = self.calculate(pred, target, self.topk, self.thrs)
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                    '`test_evaluator` fields in your config file.')

            multi_thrs = len(self.thrs) > 1
            for i, k in enumerate(self.topk):
                for j, thr in enumerate(self.thrs):
                    name = f'top{k}'
                    if multi_thrs:
                        name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                    metrics[name] = acc[i][j].item()
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            acc = self.calculate(pred, target, self.topk, self.thrs)
            metrics['top1'] = acc.item()

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        topk: Sequence[int] = (1, ),
        thrs: Sequence[Union[float, None]] = (0., ),
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        num = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).float().sum(0, keepdim=True)
            acc = correct.mul_(100. / num)
            return acc
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)

            if maxk > pred.size(1):
                raise ValueError(
                    f'Top-{maxk} accuracy is unavailable since the number of '
                    f'categories is {pred.size(1)}.')

            pred_score, pred_label = pred.topk(maxk, dim=1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            results = []
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    _correct = correct
                    if thr is not None:
                        _correct = _correct & (pred_score.t() > thr)
                    correct_k = _correct[:k].reshape(-1).float().sum(
                        0, keepdim=True)
                    acc = correct_k.mul_(100. / num)
                    results[-1].append(acc)
            return results

@WEIGHT_INITIALIZERS.register_module(name='VarPretrained')
class VarPretrainedInit:
    def __init__(self, varname:str, checkpoint, prefix=None, map_location='cpu'):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location
        self.varname = varname

    def __call__(self, moduleTop, realInit = False):#realInit默认不进行初始化，延迟手动初始化
        if not realInit:
            return
        from mmengine.runner.checkpoint import (_load_checkpoint_with_prefix,
                                                load_checkpoint,
                                                load_state_dict)
        print('VarPretrainedInit', moduleTop.__class__.__name__, self.varname)
        varargs = self.varname.split('.')
        module = moduleTop
        for m in varargs:
            module = getattr(module, m)
        print('curObj', module)
        print('VarPretrainedInit', moduleTop.__class__.__name__, self.varname, self.checkpoint)
        if self.prefix is None:
            print_log(f'load model from: {self.checkpoint}', logger='current')
            load_checkpoint(
                module,
                self.checkpoint,
                map_location=self.map_location,
                strict=False,
                logger='current')
        else:
            print_log(
                f'load {self.prefix} in model from: {self.checkpoint}',
                logger='current')
            state_dict = _load_checkpoint_with_prefix(
                self.prefix, self.checkpoint, map_location=self.map_location)
            load_state_dict(module, state_dict, strict=False, logger='current')

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: load from {self.checkpoint}'
        return info

def InitCheckpoint(moduleTop, cfg:dict):
    varname = cfg['varname']
    checkpoint = cfg['checkpoint']
    prefix = cfg.get('prefix')
    strict = cfg.get('strict', True)
    map_location='cpu'

    from mmengine.runner.checkpoint import (_load_checkpoint_with_prefix,
                                                load_checkpoint,
                                                load_state_dict)
    print('VarPretrainedInit', moduleTop.__class__.__name__, varname)
    varargs = varname.split('.')
    module = moduleTop
    for m in varargs:
        module = getattr(module, m)
    print('curObj', module)
    print('VarPretrainedInit', moduleTop.__class__.__name__, varname, checkpoint)
    if prefix is None:
        print_log(f'InitCheckpoint load model from: {checkpoint}', logger='current')
        load_checkpoint(
            module,
            checkpoint,
            map_location=map_location,
            strict=strict,
            logger='current')
    else:
        print_log(
            f'InitCheckpoint load {prefix} in model from: {checkpoint}',
            logger='current')
        state_dict = _load_checkpoint_with_prefix(
            prefix, checkpoint, map_location=map_location)
        load_state_dict(module, state_dict, strict=strict, logger='current')

    if hasattr(module, '_params_init_info'):
        info = f'{module.__class__.__name__}: load from {checkpoint}'
        update_init_info(module, init_info=info)

def DoRealInit(module, cfg):
    mod = WEIGHT_INITIALIZERS.build(cfg)
    mod(module, True)
    print('mod', mod.__class__.__name__)
    print(module)

from mmpretrain.models.losses import utils as loss_utils
@MODELS.register_module()
class ForegroundFocusLoss(nn.Module):
    def __init__(self,
                 label_smooth_val=None,
                 num_classes=None,
                 use_sigmoid=None,
                 mode='original',
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super().__init__()
        self.seg_alpha = -1
        self.pos_beta  = 1

        self.num_classes = num_classes

    def generate_one_hot_like_label(self, label):
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = loss_utils.convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]

        one_hot_likeab_lel = self.generate_one_hot_like_label(label=label)
        posIndex = one_hot_likeab_lel.to(torch.bool)

        nag = cls_score[~posIndex]
        pos = cls_score[posIndex]
        

        los_nag = F.relu(nag - self.seg_alpha)
        los_pos = F.relu(self.pos_beta - pos)
        count_nag = torch.sum(los_nag > 0).item()
        count_pos = torch.sum(los_pos > 0).item()
        if count_nag <= 0:
            count_nag = 1
        if count_pos <= 0:
            count_pos = 1
        print(__name__, cls_score.shape, label.shape, posIndex.shape, nag.shape, pos.shape, count_nag, count_pos, los_nag.max(), los_pos.max())
        loss = (torch.sum(los_nag) + torch.sum(los_pos))
        return loss
    
def BuildInvertedResidual(in_channels, mid_channels, out_channels, kernel_size=3, stride=1, wise = True):
    #in_channels = COMPRESSED_DIM
    #out_channels = in_channels
    #mid_channels = in_channels*4*2
    se_cfg = None
    if wise:
        se_cfg = dict(
            channels=mid_channels,
            ratio=4,
            act_cfg=(dict(type='ReLU'),
                        dict(
                            type='HSigmoid',
                            bias=3,
                            divisor=6,
                            min_value=0,
                            max_value=1)))
    act = 'HSwish'
    tmp = InvertedResidual(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            se_cfg=se_cfg,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            act_cfg=dict(type=act),
            #with_cp=self.with_cp
    )
    return tmp

class CNNPatchMerge(nn.Module):
    def __init__(self, in_features, midChannel, out_features=None,
                 kernel_size=3,patchHW=16):
        super().__init__()
        #self.conv = mmobj.BuildInvertedResidual(in_features, midChannel, out_features, kernel_size, 2)
        self.convReshape = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=patchHW, w=patchHW),
                                         BuildInvertedResidual(in_features, midChannel, out_features, kernel_size, 2),
                                         Rearrange('b c h w -> b (h w) c', h=patchHW//2, w=patchHW//2)
                                     )
    def forward(self, x):
        return self.convReshape(x)
        #b, n, k = x.size()
        #hw = int(math.sqrt(n))
        #x = x.transpose(1, 2).view(b, k, hw, hw)
        #x = self.conv(x)
        #out = x.view(b, x.size(1), -1).transpose(1, 2)
        ##print('CNNPatchMerge', x.shape, out.shape)
        #return out

class LeFF(nn.Module):
    def __init__(self, dim, hidden_features, patchHW,depth_kernel=3):
        super().__init__()
        #scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, hidden_features),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(hidden_features),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=patchHW, w=patchHW)
                                     )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=depth_kernel, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=patchHW, w=patchHW)
            )
        self.down_proj = nn.Sequential(nn.Linear(hidden_features, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )
    def forward(self, x):
        x = self.up_proj(x)
        #print('leff', x.shape, self.depth_conv)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
