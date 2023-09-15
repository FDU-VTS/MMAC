import argparse
import os
import time

import torch
import torchmetrics
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from datasets import MMACDataset
from preprocess import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils import AUC, Specificity,Dice,set_seed,get_model
from train_utils import get_loss,get_optimizer,get_lr_scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', type=str, default='step')
    parser.add_argument('--model', type=str, default='unet++')
    parser.add_argument('--opti', type=str, default='adam')
    parser.add_argument('--criterion', type=str, default='dice_loss')
    parser.add_argument('--lesion', type=str, default='LC')


    #model_set
    parser.add_argument('--encoder', type=str, default='resnext50_32x4d')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    set_seed(args.seed)

    # create segmentation model with pretrained encoder
    model = get_model(args)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
   
    train_dataset = MMACDataset(
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        lesion=args.lesion,
    )

    valid_dataset = MMACDataset(
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        lesion=args.lesion,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    # loss = smp.utils.losses.BCELoss()
    # if '+' in args.criterion:
    #     loss = sum([get_loss(name) for name in args.criterion.split("+")])
    # else:
    loss = get_loss(args.criterion)
    # loss = FocalLoss()
    # loss = smp.losses.DiceLoss(mode='binary')
    # loss = smp.losses.SoftBCEWithLogitsLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        Dice(threshold=0.5)
        # smp.utils.metrics.Dice(threshold=0.5),
        # AUC(threshold=0.5),
        # smp.utils.metrics.Recall(threshold=0.5),
        # Specificity(threshold=0.5)
    ]

    optimizer = get_optimizer(args.opti,args.lr,model.parameters())

    scheduler = get_lr_scheduler(args.lr_scheduler,optimizer)
    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=args.device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=args.device,
        verbose=True,
    )
    if not os.path.exists('./models/'+args.lesion):
        os.makedirs('./models/'+args.lesion)

    if not os.path.exists('./logs/'+args.lesion):
        os.makedirs('./logs/'+args.lesion)

    current_time = str(int(round(time.time() * 1000)))

    log_txt = open('./logs/'+args.lesion+'/'+current_time+'.txt', 'w')
    
    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    log_txt.write(message)
    log_txt.flush()

    max_score = 0
    best_epoch = 0

    for i in range(0, args.epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)

        valid_logs = valid_epoch.run(valid_loader)

        log_txt.write('\nEpoch: {} '.format(i))
        log_txt.write('iou_score: {} '.format(valid_logs['iou_score']))
        log_txt.write('dice_score: {} '.format(valid_logs['dice']))
        log_txt.flush()

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['dice']:
            max_score = valid_logs['dice']
            best_epoch = i
            torch.save(model, './models/'+args.lesion+'/'+current_time+'.pth')
            print('Model saved!')
        
        scheduler.step()


    log_txt.write('\nBest epoch: {} '.format(best_epoch))
    log_txt.write('Best dice_score: {} '.format(max_score))
    log_txt.flush()
    log_txt.close()
