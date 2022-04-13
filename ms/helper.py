from turtle import forward
import torch
from torch import nn
import copy
import torch.nn.functional as F
EPS = 1e-9


class EarlyStop:
    def __init__(self, pat=10) -> None:
        self._best = torch.inf
        self._count = 0
        self._pat = pat

    def step(self, metrics):
        if metrics < self._best:
            self._count = 0
            self._best = metrics
            return False
        else:
            self._count += 1
            self._best = (self._best*self._count + metrics)/(self._count + 1)
            if self._count > self._pat:
                return True
            return False


def spectral_distance(true, pred):
    true_mask = (true >= 0).float()

    pred2com = pred*true_mask
    true2com = true*true_mask

    pred2com = F.normalize(pred2com, dim=1)
    true2com = F.normalize(true2com, dim=1)

    re = torch.sum(pred2com*true2com, dim=-1)
    return torch.mean((2/torch.pi)*torch.arccos(re))


class InfoNCE(nn.Module):
    def __init__(self, temp=0.1, on_spectral=True, lambdaa=0):
        super().__init__()
        self.temp = temp
        self.on_spectral = on_spectral
        self.labmbda = lambdaa

    def forward(self, true, pred):
        B = true.shape[0]
        if self.on_spectral:
            pred_d = pred.repeat_interleave(B, 0)
            true_d = true.repeat(B, 1)
        else:
            true_d = true.repeat_interleave(B, 0)
            pred_d = pred.repeat(B, 1)
        sas = spectral_angle(true_d, pred_d)
        scores_mat = sas.reshape(B, B)
        nce = scores_mat/self.temp
        labels = torch.arange(
            start=0, end=B, dtype=torch.long, device=nce.device)
        # print([(i.item(), j.item())
        #       for i, j in zip(scores_mat[labels, labels][:10], torch.softmax(nce, dim=1)[labels, labels][:10])])
        loss_nce = F.cross_entropy(nce, labels)

        if self.labmbda > 0:
            loss_sa = spectral_distance(true, pred)
            loss = self.labmbda*loss_sa + (1 - self.labmbda)*loss_nce
            return loss, loss_nce.item(), loss_sa.item()
        else:
            return loss_nce, loss_nce.item(), 0


class FinetunePairSALoss(nn.Module):
    def __init__(self, l1_lambda=0.00005):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.act = nn.Softplus()

    def forward(self, true, pred, neg_true, neg_pred):
        l1_v = (torch.abs(pred).sum(1).mean() +
                torch.abs(neg_pred).sum(1).mean())
        sas = spectral_angle(true, pred)
        sas_neg = spectral_angle(neg_true, neg_pred)
        # base = torch.mean(self.act(sas_neg - sas))
        base = (1 - (sas-sas_neg))
        base[base < 0] = 0
        base = torch.mean(base)
        return base + self.l1_lambda*l1_v, base.item(), l1_v.item()


class FinetuneSALoss(nn.Module):
    def __init__(self, l1_lambda=0.00000, pearson=False):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.if_pearson = pearson

    def forward(self, true, pred, label):
        true_mask = (true >= 0).float()
        pred = pred*true_mask
        l1_v = torch.abs(pred).sum(1).mean()
        if not self.if_pearson:
            sas = spectral_angle(true, pred)
            base = torch.mean((sas - label)**2)
        else:
            pears = pearson_coff(true, pred)
            base = torch.mean((pears - label)**2)
        return base + self.l1_lambda*l1_v, base.item(), l1_v.item()


class FinetuneComplexSALoss(nn.Module):
    def __init__(self, l1_lambda=0.0001):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.act = nn.Sigmoid()

    def forward(self, score, label):
        sas = self.act(score)
        label[label < 0] = 0
        base = torch.mean((sas - label)**2)
        return base


class FinetuneRTLoss(nn.Module):
    def __init__(self, gap=0.1):
        super().__init__()
        self.gap = gap

    def forward(self, true, pred, label):
        l1_v = torch.abs(pred).sum(1).mean()
        sas = (true - pred)**2
        base1 = sas[label == 1].mean()
        base2 = (self.gap - sas[label == -1])
        base2[base2 < 0] = 0
        base2 = base2.mean()
        return base1 + base2, base1.item(), base2.item()


class L1Loss(nn.Module):
    def __init__(self, loss_fn, lambdaa=0.001):
        super().__init__()
        self.lambdaa = lambdaa
        self.loss_fn = loss_fn

    def forward(self, true, pred):
        true_mask = (true >= 0).float()
        pred = pred*true_mask
        l1_v = torch.abs(pred).sum(1).mean()
        base = self.loss_fn(true, pred)
        return base + self.lambdaa*l1_v, base.item(), l1_v.item()


def reshape_dims(array, MAX_SEQUENCE=30, ION_TYPES="yb", MAX_FRAG_CHARGE=3):
    n, dims = array.shape
    assert dims == 174
    return array.reshape(
        *[array.shape[0], MAX_SEQUENCE - 1,
            len(ION_TYPES), MAX_FRAG_CHARGE]
    )


def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1:, :, :] = mask
    return array


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        if charges[i] < 3:
            array[i, :, :, charges[i]:] = mask
    return array


def predict_sa(true, pred, data):
    # pred[pred < 0] = 0
    pred = pred/pred.max()
    B = pred.shape[0]
    lengths = torch.count_nonzero(data['sequence_integer'], dim=1)
    charges = torch.argmax(data["precursor_charge_onehot"], dim=1) + 1

    tide_pred = reshape_dims(pred)
    tide_pred = mask_outofrange(tide_pred, lengths)
    tide_pred = mask_outofcharge(tide_pred, charges)
    pred = tide_pred.view(B, -1)
    return spectral_angle(true, pred), pred

def predict_pearson(true, pred, data):
    pred = pred/pred.max()
    B = pred.shape[0]
    lengths = torch.count_nonzero(data['sequence_integer'], dim=1)
    charges = torch.argmax(data["precursor_charge_onehot"], dim=1) + 1

    tide_pred = reshape_dims(pred)
    tide_pred = mask_outofrange(tide_pred, lengths)
    tide_pred = mask_outofcharge(tide_pred, charges)
    pred = tide_pred.view(B, -1)
    return pearson_coff(true, pred), pred


def predict_sa_scale(true, pred, data, scale):
    pred[pred < 0] = 0
    pred = pred/pred.max()
    B = pred.shape[0]
    lengths = torch.count_nonzero(data['sequence_integer'], dim=1)
    charges = torch.argmax(data["precursor_charge_onehot"], dim=1) + 1

    tide_pred = reshape_dims(pred)
    tide_pred = mask_outofrange(tide_pred, lengths)
    tide_pred = mask_outofcharge(tide_pred, charges)
    pred = tide_pred.view(B, -1)
    return spectral_angle(true, pred, scale=scale), pred


def median_spectral_angle(true, pred):
    re = spectral_angle(true, pred)
    return torch.median(re)


def spectral_angle(true, pred, scale=None):
    true_mask = (true >= 0).float()

    pred2com = pred*true_mask
    true2com = true*true_mask

    pred2com = F.normalize(pred2com)
    if scale is None:
        true2com = F.normalize(true2com)
    else:
        true2com /= (scale+1e-9)

    re = torch.sum(pred2com*true2com, dim=-1)
    re[re > 1] = 1
    re[re < -1] = -1
    return 1 - (2/torch.pi)*torch.arccos(re)

def pearson_coff(true, pred):
    true_mask = (true >= 0).float()

    pred2com = pred*true_mask
    true2com = true*true_mask

    pred2com -= torch.mean(pred2com, dim=1).unsqueeze(-1)
    true2com -= torch.mean(true2com, dim=1).unsqueeze(-1)
    
    pred2com = F.normalize(pred2com, dim=1)
    true2com = F.normalize(true2com, dim=1)

    return torch.sum(pred2com*true2com, dim=-1)
# def spectral_angle_nonormal(true, pred):
#     true_mask = (true >= 0).float()

#     pred2com = pred*true_mask
#     true2com = true*true_mask

#     pred2com = F.normalize(pred2com)
#     true2com = F.normalize(true2com)

#     re = torch.sum(pred2com*true2com, dim=-1)
#     return - re

def mse_distance(true, pred):
    return F.mse_loss(true, pred)


def create_mask(peptides, pad_id=0):
    return (peptides == pad_id)


def set_seed(seed):
    import numpy as np
    import torch
    import random
    import pandas as pd
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
