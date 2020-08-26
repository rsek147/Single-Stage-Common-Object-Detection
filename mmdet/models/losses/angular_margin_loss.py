# ------------------------------------------------------------------------------
#  Libraries
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import weight_reduce_loss

from ..registry import LOSSES
import math


def softmax_focal_loss(logits, targets, cls_weight, gamma=1.0, eps=1e-6):
    """ Compute Focal loss by treating each class independently
        logits: value before softmax: [N x C]
        targest: one-hot vector       [N x C]
        cls_weight: weight for each classes [1 x C]
        gamma: power
        Simple implement:
            p = logits.softmax(dim=1)
            t = targets.float()
            loss = -((1-p).pow(gamma)*t*torch.log(p) + p.pow(gamma)*(1-t)*torch.log(1-p))
            loss_weight = torch.mean(loss*weight.reshape(1,-1),dim=1) #Average across classes
    """
    # Numerical stable implement
    log_p = F.log_softmax(logits)
    p = log_p.exp()
    # p = logits.softmax(dim=1)
    log_1mp = torch.log1p(eps-p)
    t = targets.float()
    loss = -(1-p).pow(gamma)*t*log_p - p.pow(gamma)*(1-t)*log_1mp

    if cls_weight is not None:
        loss = loss*cls_weight.view(1, -1)
    return loss.mean(dim=1)


# ------------------------------------------------------------------------------
#   ArcFace loss: Follow from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# ------------------------------------------------------------------------------
class AngularMarginLoss(nn.Module):
    def __init__(self, embed_channels, num_classes,
                 scale=16.0, margin=0.5, easy_margin=True, ignore_class0=True,
                 normalize_inputs=False,
                 class_weight=None, loss_weight=1.0):
        super(AngularMarginLoss, self).__init__()

        # Init weight
        self.num_classes = num_classes
        self.ignore_class0 = ignore_class0
        if self.ignore_class0:
            self.num_classes -= 1
        self.weight = nn.Parameter(torch.FloatTensor(
            self.num_classes, embed_channels))
        nn.init.xavier_uniform_(self.weight)

        # Parameters
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.ignore_class0 = ignore_class0
        self.loss_weight = loss_weight
        self.normalize_inputs = normalize_inputs

        # Auxilary constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # Cross Entropy
        self.class_weight = class_weight
        if isinstance(class_weight, list):
            self.class_weight = torch.tensor(class_weight)
        self.ce = nn.CrossEntropyLoss(
            weight=self.class_weight, reduction='none')

    def forward(self, embed_feat, labels, weight=None,
                avg_factor=None, reduction='mean', **kwargs):
        """
            embed_feat is the tensor of shape (N,C)
            labels: the class labels, of shape (N)
            weight: sample weight, of shape (N)
            avg_factor: avg_factor

        """
        # Create OneHot labels.
        sm_labels = labels.clone()
        if self.ignore_class0:
            # In Obj Detection, class0 is background.
            pos_idx = sm_labels > 0
            sm_labels = sm_labels[pos_idx]
            sm_labels -= 1
            embed_feat = embed_feat[pos_idx]

        # Compute cosine
        if self.normalize_inputs:
            embed_feat = F.normalize(embed_feat)
        W = F.normalize(self.weight)
        cosine = F.linear(embed_feat, W)
        cosine = torch.clamp(cosine, -0.99999, 0.99999)
        # Adding margin to the cosine logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        cos_phi = cosine*self.cos_m - sine*self.sin_m  # cos(theta +m)
        if self.easy_margin:
            cos_phi = torch.where(cosine > 0, cos_phi, cosine)
        else:
            cos_phi = torch.where(cosine > self.th, cos_phi, cosine - self.mm)

        one_hot = F.one_hot(sm_labels, self.num_classes)

        # Update logit based on one-hot labels
        logits = torch.where(one_hot.bool(), cos_phi, cosine)

        if self.loss_type == 'arcface':
            loss = self.ce(self.scale*logits, sm_labels)
            if self.gamma is not None:
                p = torch.exp(-loss)
                loss = (1-p)**self.gamma*loss
        elif self.loss_type == 'arcface_focal':
            loss = softmax_focal_loss(
                self.scale*logits, one_hot, None, self.gamma)
        else:
            # curricular loss
            with torch.no_grad():
                cos_p = logits.gather(dim=1, index=sm_labels[:, None])
                self.t = (1-self.momentum) * \
                    cos_p.mean() + self.momentum*self.t
                hard_neg_idx = logits > (cos_p+1e-6)
            hard_neg_val = logits*(self.t+logits)
            logits = torch.where(hard_neg_idx, hard_neg_val, logits)
            loss = self.ce(self.scale*logits, sm_labels)
            if self.loss_type == 'focal_curricular':
                with torch.no_grad():
                    pt = (-loss).exp()
                    gamma = -torch.log(torch.clamp(self.t, min=1e-5))
                    focal = (1-pt).pow(gamma)
                loss = focal*loss

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        output = dict(loss_angular=self.loss_weight*loss)

        # Compute accuracy, no impact on loss
        with torch.no_grad():
            _, y_pred = torch.max(cosine, dim=1)
            cos_pos = cosine.gather(dim=1, index=sm_labels[:, None])
            cos_neg = torch.max(torch.where(
                one_hot.bool(), -1*torch.ones_like(cosine), cosine), dim=1)[0]
            acc = y_pred.eq(sm_labels)
            acc = acc.float().mean()
            output['cw_cos_p'] = cos_pos.mean()
            output['cw_cos_n'] = cos_neg.mean()
            output['cw_class_acc'] = acc

            n_pos = sm_labels.shape[0]
            dist = 1.0 - torch.mm(embed_feat, embed_feat.t())
            mask_pos = sm_labels.expand(n_pos, n_pos).eq(
                sm_labels.expand(n_pos, n_pos).t())
            mask_neg = ~mask_pos
            eye = torch.eye(n_pos).byte().to(mask_pos)
            mask_pos[eye] = 0

            dist_p = torch.max(dist * mask_pos.float(), dim=1)[0]
            ninf = torch.ones_like(dist) * float('inf')
            dist_n = torch.min(torch.where(mask_neg, dist, ninf), dim=1)[0]

            _, top_idx = torch.topk(dist, k=2, largest=False)
            top_idx = top_idx[:, 1:]
            flat_idx = top_idx.squeeze() + n_pos * torch.arange(n_pos,
                                                                out=torch.LongTensor()).to(mask_pos)
            top1_is_same = torch.take(mask_pos, flat_idx)

            output['cw_dist_p'] = dist_p.mean()
            output['cw_dist_n'] = dist_n.mean()
            output['cw_acc'] = (dist_n > dist_p).float().mean()
            output['cw_prec'] = top1_is_same.float().mean()

        return output

    def accuracy(self, embed_feat, labels):
        with torch.no_grad():
            cosine = F.linear(F.normalize(embed_feat),
                              F.normalize(self.weight))
            _, y_pred = torch.max(cosine, dim=1)
            sm_labels = labels.clone()
            if self.ignore_class0:
                # In Obj Detection, class0 is background.
                assert torch.min(sm_labels) > 0
                sm_labels -= 1
            one_hot = F.one_hot(sm_labels, self.num_classes)

            cos_pos = cosine.gather(dim=1, index=sm_labels[:, None])
            cos_neg = torch.max(torch.where(
                one_hot.bool(), -1*torch.ones_like(cosine), cosine), dim=1)[0]

            acc = y_pred.eq(sm_labels)
            acc = acc.float().mean()

        return dict(cos_p=cos_pos.mean(), cos_n=cos_neg.mean(), emb_acc=acc)

    def infer_classes(self, embed_feat):
        with torch.no_grad():
            if self.normalize_inputs:
                embed_feat = F.normalize(embed_feat)
            W = F.normalize(self.weight)
            cosine = F.linear(embed_feat, W)

            scores = (self.scale*cosine).softmax(-1)
            _, y_pred = scores.max(dim=1)
            # if self.ignore_class0:
            #     y_pred +=1
        return y_pred


@LOSSES.register_module
class ArcFaceLoss(AngularMarginLoss):
    def __init__(self, gamma=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = 'arcface'
        self.gamma = gamma


@LOSSES.register_module
class ArcFaceFocalLoss(AngularMarginLoss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = 'arcface_focal'
        self.gamma = gamma


@LOSSES.register_module
class CurricularLoss(AngularMarginLoss):
    def __init__(self, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = 'curricular'
        self.momentum = momentum
        self.register_buffer('t', torch.zeros(1))


@LOSSES.register_module
class FocalCurricularLoss(AngularMarginLoss):
    def __init__(self, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = 'focal_curricular'
        self.momentum = momentum
        self.register_buffer('t', torch.zeros(1))
