import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.sampler import Sampler

from ..registry import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module
class SupContrastLoss(nn.Module):
    def __init__(self, in_channels, embed_channels=128,
                 scale=16.0, margin=None, easy_margin=True, ignore_class0=True,
                 normalize_inputs=False,
                 loss_weight=1.0):
        super().__init__()
        # Init weight
        self.embed_channels = embed_channels
        if embed_channels is not None:
            self.weight = nn.Parameter(
                torch.FloatTensor(embed_channels, in_channels))
            nn.init.xavier_uniform_(self.weight)
            self.use_projector = True
        else:
            self.use_projector = False

        # Parameters
        self.scale = scale
        assert margin is None or margin > 0, "Margin must be None or Positive"
        self.margin = margin
        self.easy_margin = easy_margin
        self.normalize_inputs = normalize_inputs

        # For detection, class 0 is background
        self.ignore_class0 = ignore_class0
        self.loss_weight = loss_weight

        # Auxilary constants
        if margin is not None:
            self.cos_m = math.cos(margin)
            self.sin_m = math.sin(margin)
            self.th = math.cos(math.pi - margin)
            self.mm = math.sin(math.pi - margin) * margin

    def pair_similarity(self, feat_1, feat_2, mask_pos=None, mask_neg=None):
        cosine = torch.mm(feat_1, feat_2.t())
        cosine = torch.clamp(cosine, -0.99999, 0.99999)

        if self.margin is not None and mask_pos is not None:
            # Adding margin to the cosine logits
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            cos_phi = cosine*self.cos_m - sine*self.sin_m  # cos(theta +m)
            if self.easy_margin:
                cos_phi = torch.where(cosine > 0, cos_phi, cosine)
            else:
                cos_phi = torch.where(
                    cosine > self.th, cos_phi, cosine - self.mm)
            cosine = torch.where(mask_pos, cos_phi, cosine)

        return self.scale*cosine

    def forward(self, inputs, targets, weight=None,
                avg_factor=None, reduction='mean', **kwargs):
        # embed_feat=F.normalize(F.linear(inputs,self.weight))
        embed_feat = inputs
        if self.normalize_inputs:
            embed_feat = F.normalize(embed_feat)

        if self.use_projector:
            embed_feat = F.linear(embed_feat, F.normalize(self.weight))
            # embed_feat = F.normalize(embed_feat)

        n = inputs.size(0)

        # Class 0 is background
        p_idx = targets > 0
        n_pos = p_idx.sum()

        # We compare labels between foreground first
        targets_pos = targets[p_idx]
        mask_pos = targets_pos.expand(n_pos, n_pos).eq(
            targets_pos.expand(n_pos, n_pos).t())
        mask_neg = ~mask_pos
        # The diagonal element (compare with itself) is excluded
        eye = torch.eye(n_pos).byte().to(mask_pos)
        mask_pos[eye] = 0

        # Compute similarity
        x_obj = embed_feat[p_idx, :]
        z = self.pair_similarity(x_obj, x_obj, mask_pos, mask_neg)

        # Adding background
        if not self.ignore_class0:
            z_bg = self.pair_similarity(x_obj, embed_feat[~p_idx, :])
            z = torch.cat([z, z_bg], dim=1)

            # Adding background
            mask_bg = torch.zeros(
                (n_pos, n-n_pos), dtype=torch.bool).to(mask_pos)
            mask_pos = torch.cat([mask_pos, mask_bg], dim=1)
            mask_neg = torch.cat([mask_neg, ~mask_bg], dim=1)
            mask_rest = ~torch.cat([eye, mask_bg], dim=1)
        else:
            mask_rest = ~eye

        # Loss
        loss_num = -(z*mask_pos).sum(dim=1)
        loss_den = ((z.exp()*mask_rest).sum(dim=1) + 1e-6).log()

        # Num positive pairs for each anchor
        num_pos_pairs = mask_pos.sum(dim=1)
        valid_anchors = num_pos_pairs > 0
        num_pos_pairs = num_pos_pairs[valid_anchors]
        loss_num = loss_num[valid_anchors]/num_pos_pairs
        loss_den = loss_den[valid_anchors]

        loss = loss_num + loss_den

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()

        # loss = weight_reduce_loss(
        # 	loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        loss = weight_reduce_loss(
            loss, weight=None, reduction=reduction, avg_factor=avg_factor)

        # Avoid Nan loss
        if loss != loss:
            loss = 0*loss_den.sum()

        output = dict(loss_contrastive=self.loss_weight*loss)
        # calculate metrics, no impact on loss
        with torch.no_grad():
            dist = 1 - torch.mm(x_obj, x_obj.t())
            # dist = 1 - z/self.scale
            dist_p = torch.max(dist * mask_pos.float(), dim=1)[0]
            ninf = torch.ones_like(dist) * float('inf')
            dist_n = torch.min(torch.where(mask_neg, dist, ninf), dim=1)[0]

            _, top_idx = torch.topk(dist, k=2, largest=False)
            top_idx = top_idx[:, 1:]
            flat_idx = top_idx.squeeze() + n_pos * torch.arange(
                n_pos, out=torch.LongTensor()).to(mask_pos)
            top1_is_same = torch.take(mask_pos, flat_idx)

            output['pw_dist_p'] = dist_p.mean()
            output['pw_dist_n'] = dist_n.mean()
            output['pw_acc'] = (dist_n > dist_p).float().mean()
            output['pw_prec'] = top1_is_same.float().mean()

        return output


@LOSSES.register_module
class SupContrastNegLoss(SupContrastLoss):
    def forward(self, inputs, targets, weight=None,
                avg_factor=None, reduction='mean', **kwargs):
        # embed_feat=F.normalize(F.linear(inputs,self.weight))
        embed_feat = inputs
        if self.normalize_inputs:
            embed_feat = F.normalize(embed_feat)

        if self.use_projector:
            embed_feat = F.linear(embed_feat, F.normalize(self.weight))
            # embed_feat = F.normalize(embed_feat)

        n = inputs.size(0)

        # Class 0 is background
        p_idx = targets > 0
        n_pos = p_idx.sum()

        # We compare labels between foreground first
        targets_pos = targets[p_idx]
        mask_pos = targets_pos.expand(n_pos, n_pos).eq(
            targets_pos.expand(n_pos, n_pos).t())
        mask_neg = ~mask_pos
        # The diagonal element (compare with itself) is excluded
        eye = torch.eye(n_pos).byte().to(mask_pos)
        mask_pos[eye] = 0

        # Compute similarity
        x_obj = embed_feat[p_idx, :]
        logits = self.pair_similarity(x_obj, x_obj, mask_pos, mask_neg)

        # Adding background
        if not self.ignore_class0:
            logits_bg = self.pair_similarity(x_obj, embed_feat[~p_idx, :])
            logits = torch.cat([logits, logits_bg], dim=1)

            # Adding background
            mask_bg = torch.zeros(
                (n_pos, n-n_pos), dtype=torch.bool).to(mask_pos)
            mask_pos = torch.cat([mask_pos, mask_bg], dim=1)
            mask_neg = torch.cat([mask_neg, ~mask_bg], dim=1)

        # Loss
        exp_logits = logits.exp()
        # sum of exp(cos(negative_pairs))
        neg_sum = (exp_logits*mask_neg).sum(dim=1)
        # Each positive will be added with sum of negative
        den = neg_sum[:, None] + exp_logits
        # Get numerator for positive pairs
        pos_num = torch.masked_select(logits, mask_pos)
        # Get log(denominator) for positive pairs
        pos_den = torch.masked_select(den, mask_pos).log()
        num_pos_pairs = mask_pos.sum(dim=1, keepdim=True).repeat(
            1, logits.shape[1])  # Number of positive pairs for each anchor
        num_pos_pairs = torch.masked_select(num_pos_pairs, mask_pos).float()

        valid_anchors = num_pos_pairs > 0
        loss = (pos_den[valid_anchors] - pos_num[valid_anchors]
                )/num_pos_pairs[valid_anchors]

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()

        loss = weight_reduce_loss(
            loss, weight=None, reduction=reduction, avg_factor=avg_factor)
        # loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        # Avoid Nan loss
        if loss != loss:
            import pdb
            pdb.set_trace()
            loss = 0*loss

        output = dict(loss_contrastive=self.loss_weight*loss)
        # calculate metrics, no impact on loss
        with torch.no_grad():
            dist = 1 - torch.mm(x_obj, x_obj.t())
            # dist = 1 - z/self.scale
            dist_p = torch.max(dist * mask_pos.float(), dim=1)[0]
            ninf = torch.ones_like(dist) * float('inf')
            dist_n = torch.min(torch.where(mask_neg, dist, ninf), dim=1)[0]

            _, top_idx = torch.topk(dist, k=2, largest=False)
            top_idx = top_idx[:, 1:]
            flat_idx = top_idx.squeeze() + n_pos * torch.arange(
                n_pos, out=torch.LongTensor()).to(mask_pos)
            top1_is_same = torch.take(mask_pos, flat_idx)

            output['pw_dist_p'] = dist_p.mean()
            output['pw_dist_n'] = dist_n.mean()
            output['pw_acc'] = (dist_n > dist_p).float().mean()
            output['pw_prec'] = top1_is_same.float().mean()

        return output


@LOSSES.register_module
class CurContrastLoss(SupContrastLoss):
    def __init__(self, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.register_buffer('t', torch.zeros(1))

    def pair_similarity(self, feat_1, feat_2, mask_pos=None, mask_neg=None):
        cosine = torch.mm(feat_1, feat_2.t())
        cosine = torch.clamp(cosine, -0.99999, 0.99999)

        if self.margin is not None and mask_pos is not None:
            # Adding margin to the cosine logits
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            cos_phi = cosine*self.cos_m - sine*self.sin_m  # cos(theta +m)
            if self.easy_margin:
                cos_phi = torch.where(cosine > 0, cos_phi, cosine)
            else:
                cos_phi = torch.where(
                    cosine > self.th, cos_phi, cosine - self.mm)

            cosine = torch.where(mask_pos, cos_phi, cosine)

            # Reduce cosine of hard negative samples
            with torch.no_grad():
                # Select the positive pair that have min cosine
                min_cos_phi, _ = torch.where(
                    mask_pos, cos_phi, torch.ones_like(cosine)).min(dim=1)
                self.t = (1-self.momentum) * \
                    min_cos_phi.mean() + self.momentum*self.t
                hard_neg_idx = cosine > min_cos_phi
            hard_neg_val = cosine*(self.t+cosine)
            cosine = torch.where(hard_neg_idx*mask_neg, hard_neg_val, cosine)

        return self.scale*cosine


@LOSSES.register_module
class CurContrastNegLoss(SupContrastNegLoss):
    def __init__(self, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.register_buffer('t', torch.zeros(1))

    def pair_similarity(self, feat_1, feat_2, mask_pos=None, mask_neg=None):
        cosine = torch.mm(feat_1, feat_2.t())
        cosine = torch.clamp(cosine, -0.99999, 0.99999)

        if self.margin is not None and mask_pos is not None:
            # Adding margin to the cosine logits
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            cos_phi = cosine*self.cos_m - sine*self.sin_m  # cos(theta +m)
            if self.easy_margin:
                cos_phi = torch.where(cosine > 0, cos_phi, cosine)
            else:
                cos_phi = torch.where(
                    cosine > self.th, cos_phi, cosine - self.mm)

            cosine = torch.where(mask_pos, cos_phi, cosine)

            # Reduce cosine of hard negative samples
            with torch.no_grad():
                # Select the positive pair that have min cosine
                min_cos_phi, _ = torch.where(
                    mask_pos, cos_phi, torch.ones_like(cosine)).min(dim=1)
                self.t = (1-self.momentum) * \
                    min_cos_phi.mean() + self.momentum*self.t
                hard_neg_idx = cosine > min_cos_phi
            hard_neg_val = cosine*(self.t+cosine)
            cosine = torch.where(hard_neg_idx*mask_neg, hard_neg_val, cosine)

        return self.scale*cosine
