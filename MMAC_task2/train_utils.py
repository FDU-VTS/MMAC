import segmentation_models_pytorch as smp
import torch
import sys
import numpy as np
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils.meter import AverageValueMeter
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LambdaLR


def get_loss(name):
    if name == 'dice_loss':
        loss = smp.utils.losses.DiceLoss()
        return loss
    elif name == 'bce_loss':
        return smp.losses.SoftBCEWithLogitsLoss()
    elif name == 'focal_loss':
        loss = smp.losses.FocalLoss(mode='binary')
        loss.__name__ = 'FocalLoss'
        return loss
    elif name == 'tversky_loss':
        loss = smp.losses.TverskyLoss(mode='binary')
        loss.__name__ = 'TverskyLoss'
        return loss
    elif name == 'dice_loss+bce_loss':
        return smp.utils.losses.DiceLoss()+smp.utils.losses.BCELoss()
    elif name == 'dice_loss+focal_loss':
        return smp.utils.losses.DiceLoss()+FocalLoss()
    else:
        return None

def get_optimizer(name,lr,params):
    if name == 'adam':
        return torch.optim.Adam([
            dict(params=params, lr=lr), # origin 0.0001
        ])
    elif name == 'adamw':
        return torch.optim.AdamW(
            params=params, lr=lr, weight_decay=1e-3
            )
    else: 
        return None

def get_lr_scheduler(name,optimizer):
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)
    elif name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-9)
    elif name == 'warmup_cosine':
        return CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=2, eta_max=0.1, T_warmup=10)
    else:
        return None

class CosineAnnealingWarmUpRestarts(LambdaLR):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.cycle_count = 0
        self.cycle_iterations = 0
        self.total_iterations = 0
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def get_lr(self,):
        return [group['lr'] for group in self.optimizer.param_groups]

    def lr_lambda(self, step):
        if self.total_iterations == 0 or step == 0:
            return 1.0
        elif step <= self.T_warmup:
            return step / self.T_warmup
        else:
            step = step - self.T_warmup
            cycle_length = self.T_0 * (self.T_mult ** self.cycle_count)
            if step >= cycle_length:
                self.cycle_count += 1
                self.cycle_iterations = 0
                self.T_0 *= self.T_mult
                return self.eta_max
            else:
                self.cycle_iterations += 1
                return 0.5 * (math.cos(math.pi * self.cycle_iterations / cycle_length) + 1) * self.eta_max

        self.total_iterations += 1
        return lr_lambda


### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()
        if len(rgb_noisy.shape) == 5:
            lam_nosie = lam.unsqueeze(-1)
        else:
            lam_nosie = lam

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam_nosie * rgb_noisy + (1-lam_nosie) * rgb_noisy2

        return rgb_gt, rgb_noisy
    
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()
    
    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        if type(prediction) == type(()):
            prediction = prediction[0]
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction

class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            if type(prediction) == type(()):
                prediction = prediction[0]
            loss = self.loss(prediction, y)
        return loss, prediction


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

class FocalLoss(base.Loss, focal_loss):
    pass