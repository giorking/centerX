import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import scipy.ndimage as nd
from ..utils.utils import sim_dis_compute

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor) #int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept)-1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())
        return new_target

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)

class CriterionAdditionalGP(nn.Module):
    def __init__(self, D_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.D = D_net
        self.lambda_gp = lambda_gp

    def forward(self, d_in_S, d_in_T):
        assert d_in_S[0].shape == d_in_T[0].shape,'the output dim of D with teacher and student as input differ'

        real_images = d_in_T[0]
        fake_images = d_in_S[0]
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0],
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(out[0].size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss

class CriterionAdvForG(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_S_no_use):
        g_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return g_loss_fake

class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        assert d_out_S[0].shape == d_out_T[0].shape,'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2*0.4

class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2*0.4

class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        preds_T[0].detach()
        assert preds_S[0].shape == preds_T[0].shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S[0].shape
        softmax_pred_T = F.softmax(preds_T[0].permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S[0].permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_ind, single_scale=False):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale
        self.single_scale = single_scale

    def forward(self, preds_S, preds_T):
        if self.single_scale:
            feat_S = preds_S
            feat_T = preds_T
        else:
            feat_S = preds_S[self.feat_ind]
            feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss

class CriterionKD_old(nn.Module):
    def __init__(self, ignore_index=255, upsample=False, use_weight=True, T=1, sp=0, pp=0):
        super(CriterionKD_old, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.upsample = upsample
        self.soft_p = sp
        self.pred_p = pp
        self.T = T
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds, soft):
        h, w = soft[self.soft_p].size(2), soft[self.soft_p].size(3)
        if self.upsample:
            scale_pred = F.upsample(input=preds[self.pred_p], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        else:
            scale_pred = preds[self.pred_p]
        scale_soft = F.upsample(input=soft[self.soft_p], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        loss2 = self.criterion_kd(F.log_softmax(scale_pred / self.T, dim=1), F.softmax(scale_soft / self.T, dim=1))
        return loss2

class CriterionKD(nn.Module):
    '''
    '''
    def __init__(self,ignore_label=255):
        super(CriterionKD,self).__init__()
        self.ignore_label = ignore_label
        self.criterion_kd = nn.KLDivLoss()
    def forward(self,predict,target,weight=None):

        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1}".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1}".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1}".format(predict.size(3), target.size(3))
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict[target_mask]
        loss = self.criterion_kd(F.log_softmax(predict),F.log_softmax(target))
        return loss

class Cos_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #x = x.view(1,1,)
        C, width, height = x.size()
        x = x.view(-1,C,width,height)
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention

class CriterionSDcos(nn.Module):
    '''
    structure distillation loss based on graph
    '''
    def __init__(self, ignore_index=255, use_weight=True, pp=1, sp=1):
        super(CriterionSDcos, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.soft_p = sp
        self.pred_p = pp
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds[self.pred_p])
        graph_t = self.attn(soft[self.soft_p])
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph

class CriterionKlDivergence(nn.Module):
    '''
    '''
    def __init__(self):
        super(CriterionKlDivergence,self).__init__()
        self.criterion_kd = nn.KLDivLoss()
    def forward(self,s_feature,t_feature):
        #assert not t_feature.requires_grad
        assert s_feature.dim() == 4
        assert t_feature.dim() == 4
        assert s_feature.size(0) == t_feature.size(0),'{0} vs {1}'.format(s_feature.size(0),t_feature.size(0))
        assert s_feature.size(2) == t_feature.size(2),'{0} vs {1}'.format(s_feature.size(2),t_feature.size(2))
        assert s_feature.size(3) == t_feature.size(3),'{0} vs {1}'.format(s_feature.size(3),t_feature.size(3))
        return dict(loss = self.criterion_kd(F.log_softmax(s_feature),F.softmax(t_feature)))