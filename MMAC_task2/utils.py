import matplotlib.pyplot as plt

# helper function for data visualization
import numpy as np
import torch
import torchmetrics
import cv2
import torch.nn as nn
import random
import segmentation_models_pytorch as smp

from scipy import ndimage
from segmentation_models_pytorch.utils import base,functional
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils.functional import _take_channels, _threshold
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def roc_auc_score(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate auc score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    auroc = torchmetrics.AUROC(average='macro', num_classes=1)
    auroc(pr.view(-1), gt.type(torch.uint8).view(-1))
    auc = auroc.compute()
    return auc

class AUC(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return roc_auc_score(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

def specificity(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate specificity score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: specificity score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    tn = gt.view(-1).shape[0] - tp - fp -fn

    score = (tn + eps) / (tn + fp + eps)

    return score


class Specificity(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return specificity(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

def dice(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    dice_eff = ((2. * intersection) + eps) / (torch.sum(gt) + torch.sum(pr) + eps)
    return dice_eff

class Dice(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return dice(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
# def get_IoU(gt, pred, classId=1):
#     if np.sum(gt) == 0:
#         return np.nan
#     else:
#         intersection = np.logical_and(gt == classId, pred == classId)
#         union = np.logical_or(gt == classId, pred == classId)
#         iou = np.sum(intersection) / np.sum(union)
#         return iou


class focal_loss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=1, eps=1e-7):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class FocalLoss(focal_loss, base.Loss):
    pass

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    #cudnn.benchmark = True       
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args):
    if args.model == 'unet++':
        return smp.UnetPlusPlus(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
    elif args.model == 'unet':
        return smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
    elif args.model == 'manet':
        return smp.MAnet(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
    elif args.model == 'deeplabv3+':
        return smp.DeepLabV3Plus(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
    elif args.model == 'deeplabv3':
        return smp.DeepLabV3(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
    elif args.model == 'fpn':
        return smp.FPN(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
