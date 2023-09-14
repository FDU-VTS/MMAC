import numpy as np
import torch
import argparse
import os
import torch.nn as nn
from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from dataset import grading_dataset
from datetime import datetime
from functions import progress_bar
import torch.nn.functional as F
from sklearn.metrics import f1_score,roc_auc_score,cohen_kappa_score,accuracy_score,confusion_matrix,r2_score, mean_absolute_error
from imblearn.metrics import sensitivity_score, specificity_score
from loss import FocalLoss
from lr_scheduler import LRScheduler
import random
import torch.backends.cudnn as cudnn
import timm
from threshold import OptimizedRounder, OptimizedF1
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import utils

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='res50', help='model')
parser.add_argument('--visname', '-vis', default='kaggle', help='visname')
parser.add_argument('--batch-size', '-bs', default=32, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-4, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=5, type=int, help='n-classes')
parser.add_argument('--task', '-task', default='clf', type=str, help='clf/ref')
parser.add_argument('--pretrained', '-pre', default=False, type=bool, help='use pretrained model')
parser.add_argument('--dataset', '-data', default='pair', type=str, help='dataset')
parser.add_argument('--test', '-test', default=False, type=bool, help='test mode')
parser.add_argument('--KK', '-KK', default=0, type=int, help='KFold')
parser.add_argument('--clf', '-clf', default='5', type=str, help='5')
parser.add_argument('--val_certain_epoch', '-val', type=bool, default=False)
parser.add_argument('--optimizer', '-optimizer', default='adam', type=str, help='adam, sgd')
parser.add_argument('--loss', '-loss', default='ce', type=str, help='ce, focal')
parser.add_argument('--challenge', '-challenge', default=1, type=int, help='1')

parser.add_argument('--ddr_pseudo', '-ddr_pseudo', default=False, type=bool, help='pseudo')
parser.add_argument('--ddr_pseudo_r', '-ddr_pseudo_r', default=0.1, type=float, help='r')


parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)
parser.add_argument("--lambda_value", default=0.25, type=float)

val_epoch = 1
test_epoch = 1


my_whole_seed = 0
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False 



def parse_args():
    global args
    args = parser.parse_args()

parse_args()

best_acc = 0
best_kappa = 0
if args.challenge == 1:
    best_average_score = 0
    best_average_score_opt = 0

best_test_acc = 0
best_test_kappa = 0


def main_single():
    
    global best_kappa
    global best_average_score
    global best_average_score_opt
    global save_dir
    
    parse_args()

    if args.task == 'clf':
        n_c = args.n_classes
    elif args.task == 'reg':
        n_c = 1

    if args.model == 'resnet50':
        # net = timm.create_model('resnet50', pretrained=True, num_classes=n_c)
        net = utils.get_model('resnet50',num_class=n_c)
    elif args.model == 'resnet18':
        net = utils.get_model('resnet18',num_class=n_c)
    elif args.model == 'resnet34':
        net = utils.get_model('resnet34',num_class=n_c)
    elif args.model == 'vgg16':
        net = utils.get_model('vgg16',num_class=n_c)
    elif args.model == 'inceptionv3':
        net = utils.get_model('inceptionv3',num_class=n_c)
    elif args.model == 'densenet121':
        net = utils.get_model('densenet121',num_class=n_c)
    elif args.model == 'vitb16':
        net = utils.get_model('vitb16',num_class=n_c)
    elif args.model == 'vitl16':
        net = utils.get_model('vitl16',num_class=n_c)
    elif args.model == 'swint':
        net = utils.get_model('swint',num_class=n_c)
    elif args.model == 'swins':
        net = utils.get_model('swins',num_class=n_c)
    elif args.model == 'swinb':
        net = utils.get_model('swinb',num_class=n_c)
    elif args.model == 'uni4eye':
        net = utils.get_model('uni4eye',num_class=n_c)
    elif args.model == 'ssit':
        net = utils.get_model('ssit',num_class=n_c)
    elif args.model == 'swins':
        net = utils.get_model('swins',num_class=n_c)
    elif args.model == 'convnext_t':
        net = utils.get_model('convnext_t',num_class=n_c)
    elif args.model == 'convnext_s':
        net = utils.get_model('convnext_s',num_class=n_c)
    elif args.model == 'convnext_b':
        net = utils.get_model('convnext_b',num_class=n_c)
    elif args.model == 'convnext_l':
        net = utils.get_model('convnext_l',num_class=n_c)
    elif args.model == 'efficientnetb6':
        net = utils.get_model('efficientnetb6',num_class=n_c)
    elif args.model == 'efficientnetb0':
        net = utils.get_model('efficientnetb0',num_class=n_c)
    elif args.model == 'efficientnetb1':
        net = utils.get_model('efficientnetb1',num_class=n_c)
    elif args.model == 'efficientnetb2':
        net = utils.get_model('efficientnetb2',num_class=n_c)
    elif args.model == 'efficientnetb3':
        net = utils.get_model('efficientnetb3',num_class=n_c)
    elif args.model == 'efficientnetb4':
        net = utils.get_model('efficientnetb4',num_class=n_c)
    elif args.model == 'efficientnetb5':
        net = utils.get_model('efficientnetb5',num_class=n_c)
    elif args.model == 'efficientnetb7':
        net = utils.get_model('efficientnetb7',num_class=n_c)
    elif args.model == 'seresnext50':
        net = utils.get_model('seresnext50',num_class=n_c)
    elif args.model == 'squeezenet1_0':
        net = utils.get_model('squeezenet1_0',num_class=n_c)
    elif args.model == 'biomedgpt_tiny':
        net = utils.get_model('biomedgpt_tiny',num_class=n_c)
    elif args.model == 'biomedgpt_medium':
        net = utils.get_model('biomedgpt_medium',num_class=n_c)
    elif args.model == 'pmcclip512':
        net = utils.get_model('pmcclip512',num_class=n_c)
    elif args.model == 'pmcclip224':
        net = utils.get_model('pmcclip224',num_class=n_c)
    elif args.model == 'pubmedclip':
        net = utils.get_model('pubmedclip',num_class=n_c)
    elif args.model == 'lesion_based_resnet50':
        net = utils.get_model('lesion_based_resnet50',num_class=n_c)
    elif args.model == 'xm_rotation_resnet18':
        net = utils.get_model('xm_rotation_resnet18',num_class=n_c)
    elif args.model == 'xm_ssl_resnet18':
        net = utils.get_model('xm_ssl_resnet18',num_class=n_c)
    elif args.model == 'resnet50_ddrpre':
        net = utils.get_model('resnet50_ddrpre',num_class=n_c)
    elif args.model == 'resnet50_ddrprew':
        net = utils.get_model('resnet50_ddrprew',num_class=n_c)
    elif args.model == 'resnet50_kagglepre':
        net = utils.get_model('resnet50_kagglepre',num_class=n_c)
    elif args.model == 'mobilev3':
        net = utils.get_model('mobilev3',num_class=n_c)
    print(net)


    net = nn.DataParallel(net)
    net = net.cuda()

    if args.challenge == 1:
        trainset = grading_dataset(train=True, val=False, test=False, KK=args.KK, private_data=args.private_data, ddr_pseudo=args.ddr_pseudo, ddr_pseudo_r=args.ddr_pseudo_r)
        valset = grading_dataset(train=False, val=True, test=False, KK=args.KK, private_data=args.private_data, ddr_pseudo=args.ddr_pseudo, ddr_pseudo_r=args.ddr_pseudo_r)

    drop_last=False
    trainloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=drop_last)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # optim & crit
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #1e-5

    lr_scheduler = LRScheduler(optimizer, len(trainloader), args)

    if args.task == 'clf':
        if args.challenge == 1:
            # weight = torch.tensor([0.0502, 0.0492, 0.0906, 0.3381, 0.4718]) # mmac wloss
            # weight = torch.tensor([1.,1.,1.,1.,2.]) # mmac wloss1
            # weight = torch.tensor([1.,1.,1.,1.,5.]) # mmac wloss2
            weight = None
        
        if args.loss == 'ce':
            criterion = nn.CrossEntropyLoss(weight=weight)
        elif args.loss == 'focal':
            criterion = FocalLoss(class_num=n_c)
    elif args.task == 'reg':
        # criterion = nn.SmoothL1Loss()
        # criterion = nn.MAELoss()
        criterion = nn.MSELoss()

    criterion = criterion.cuda()

    if args.challenge == 1:
        file = 'challenge1/'
    save_dir = './checkpoints/' + file + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+file + args.visname+'.txt','a')   

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        net.train()
        total_loss = .0
        total = .0
        correct = .0
        count = .0

        predicted_list = []
        pred_list = []
        pred_f1opt_list = []
        label_list = []

        for i, (x, label, id) in enumerate(trainloader):
            lr = lr_scheduler.update(i, epoch)
            x = x.float().cuda()
            label = label.cuda()
            
            y_pred = net(x)

            predicted_list.extend(y_pred.squeeze(-1).cpu().detach())

            if args.task == 'clf':
                if args.model == 'inceptionv3':
                    loss_clf = 0.5 * (criterion(y_pred[0], label) + criterion(y_pred[1], label))
                    prediction = y_pred[0].max(1)[1]

                else:
                    loss_clf = criterion(y_pred, label)
                    prediction = y_pred.max(1)[1]

            elif args.task == 'reg':
                loss_clf = criterion(y_pred, label.unsqueeze(1).float())
                prediction = y_pred


            loss = loss_clf
            total_loss += loss.item()
            total += x.size(0)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if args.challenge == 1 and args.task == 'reg':
                prediction = nor_pre(prediction)

            count += 1    

            pred_list.extend(prediction.cpu().detach())      
            label_list.extend(label.cpu().detach())

            progress_bar(i, len(trainloader), 'Loss: %.3f '
                              % (total_loss / (i + 1)))
        
        if args.challenge == 1:
            if args.task == 'clf':
                kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
                f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                specificity = specificity_score(np.array(label_list), np.array(pred_list), average='macro')
                average_score = (kappa + f1 + specificity) / 3

                test_log.write('Epoch:%d lr:%.5f Loss:%.4f Kappa:%.4f F1:%.4f Spe:%.4f Avg:%.4f \n'%(epoch, lr, total_loss / count, kappa, f1, specificity, average_score))
                test_log.flush() 

                if average_score >= best_average_score:
                    print('Saving..')
                    state = {
                        'net': net.state_dict(),
                    }
                    save_name = os.path.join(save_dir, str(epoch) + '.pkl')
                    torch.save(state, save_name)
                    best_average_score = average_score

            elif args.task == 'reg':
                # normal threshold
                kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
                f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                specificity = specificity_score(np.array(label_list), np.array(pred_list), average='macro')
                average_score = (kappa + f1 + specificity) / 3       
                # optimize threshold
                optR = OptimizedRounder()
                optR.fit(predicted_list, np.array(label_list))
                coefficients = optR.coefficients()
                optimized_predictions = optR.predict(np.array(predicted_list), coefficients)

                kappa_opt = cohen_kappa_score(np.array(label_list), optimized_predictions, weights='quadratic')
                f1_opt = f1_score(np.array(label_list), optimized_predictions, average='macro')
                specificity_opt = specificity_score(np.array(label_list), optimized_predictions, average='macro')
                average_score_opt = (kappa_opt + f1_opt + specificity_opt) / 3         

                test_log.write('Epoch:%d lr:%.5f Loss:%.4f Kappa:%.4f F1:%.4f Spe:%.4f Avg:%.4f | Opt Kappa:%.4f F1:%.4f Spe:%.4f Avg:%.4f  coef:%s \n'%(epoch, lr, total_loss / count, kappa, f1, specificity, average_score, kappa_opt, f1_opt, specificity_opt, average_score_opt, str(coefficients)))
                test_log.flush() 
                
                if average_score_opt >= best_average_score_opt:
                    print('Saving..')
                    state = {
                        'net': net.state_dict(),
                    }
                    save_name = os.path.join(save_dir, str(epoch) + '.pkl')
                    torch.save(state, save_name)
                    best_average_score_opt = average_score_opt

        # if (epoch+1)%val_epoch == 0:
        #     main_single_val(net, valloader, epoch, test_log, optimizer)


@torch.no_grad()
def main_single_val(net, valloader, epoch, test_log, optimizer):
    global best_acc
    global best_kappa
    global best_average_score

    net = net.eval()
    correct = .0
    total = .0
    count = .0

    pred_list = []
    predicted_list = []
    label_list = []

    for i, (x, label, id) in enumerate(valloader):
        x = x.float().cuda()
        label = label.cuda()

        y_pred = net(x)

        predicted_list.extend(y_pred.squeeze(-1).cpu().detach())
        label_list.extend(label.cpu().detach())
        if args.task == 'clf':
            pred = y_pred.max(1)[1]
            pred_list.extend(pred.cpu().detach())
        elif args.task == 'reg':
            pred = nor_pre(y_pred)
            pred_list.extend(pred.cpu().detach())

        progress_bar(i, len(valloader))


    if args.task == 'clf':
        kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
        f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')

        cm1 = confusion_matrix(np.array(label_list).tolist(), np.array(pred_list).tolist())
        specificity = specificity_score(np.array(label_list), np.array(pred_list), average='macro')
        
        average_score = (kappa + f1 + specificity) / 3       

        print('val epoch:', epoch, 'kappa: ', kappa, ' f1: ', f1, ' spe: ', specificity, 'con: ', str(cm1))
        test_log.write('Val Epoch:%d   Kappa:%.4f  F1:%.4f  Spe:%.4f  Avg:%.4f  con:%s \n'%(epoch, kappa, f1, specificity, average_score, str(cm1)))
        test_log.flush()

    elif args.task == 'reg':
        # normal threshold
        kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
        acc = accuracy_score(np.array(label_list), np.array(pred_list))       
        # optimize threshold
        optR = OptimizedRounder()
        optR.fit(predicted_list, np.array(label_list))
        coefficients = optR.coefficients()
        optimized_predictions = optR.predict(np.array(predicted_list), coefficients)

        kappa_opt = cohen_kappa_score(np.array(label_list), optimized_predictions, weights='quadratic')
        acc_opt = accuracy_score(np.array(label_list), optimized_predictions)

        print('val epoch:', epoch, 'acc: ', acc, 'kappa: ', kappa)
        test_log.write('Val Epoch:%d   Accuracy:%.4f   kappa:%.4f  Accuracy_opt:%.4f   kappa_opt:%.4f  coef: %s\n'%(epoch,acc, kappa,acc_opt, kappa_opt, str(coefficients)))
        test_log.flush()  


    if average_score >= best_average_score:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            # 'kappa': kappa,
            # 'epoch': epoch,
            # 'optimizer': optimizer
        }
        save_name = os.path.join(save_dir, str(epoch) + '.pkl')
        torch.save(state, save_name)
        best_average_score = average_score


if __name__ == '__main__':
    parse_args()
    if args.test:
        main_single_test()
    else:
        main_single()
