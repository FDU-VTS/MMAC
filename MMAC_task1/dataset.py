import os
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T 
from PIL import Image
import torch
import csv
from random import shuffle, sample
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from sklearn.model_selection import KFold

class grading_dataset(data.Dataset):
    def __init__(self, train=False, val=False, test=False, test_tta=False, clf=5, all=False, KK=0, supcon=False, lanet=False, ddr_pseudo=False, ddr_pseudo_r=0.1):
        self.train = train
        self.val = val
        self.test = test
        self.clf = clf
        self.path = 'data/1. Classification of Myopic Maculopathy/'
        self.supcon = supcon
        self.lanet = lanet

        if train or val or all:
            self.file = '1. Images/1. Training Set/'
            e_file = '2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv'

        self.imgs = []
        img_list = [[] for _ in range(clf)]

        if test or test_tta:
            for i in range(len(os.listdir(self.file))):
                x = os.listdir(self.file)[i]
                self.imgs.append([self.file+x, -1, x])

        elif train or val:
            csv_file = pd.read_csv(self.path + e_file)
            self.dict_label = {}
            for index, row in csv_file.iterrows():
                image_id = row['image']
                rank = int(row['myopic_maculopathy_grade'])
                img_list[rank].append(image_id)
                
                self.imgs.append([self.path+self.file+image_id, rank, image_id])

                #resample
                # if rank == 4:
                #     for _ in range(5-1):
                #         self.imgs.append([self.path+self.file+image_id, rank, image_id])


            # classes = range(clf)
            # for i in classes:
            #     # print("CV:",KK)
            #     kf = KFold(n_splits=5,shuffle=True,random_state=5)
            #     for kk, (a,b) in enumerate(kf.split(range(len(img_list[i])))):
            #         if kk == KK:
            #             train_index, val_index = a, b
            #             if self.train:
            #                 print("Grade",i, ':',  len(train_index),len(val_index))
            #     if train:
            #         for index in train_index:
            #             x = img_list[i][index]
            #             self.imgs.append([self.path+self.file+x, i, x])
            #     else:
            #         for index in val_index:
            #             x = img_list[i][index]
            #             self.imgs.append([self.path+self.file+x, i, x])       
        # elif all:
        #     csv_file = pd.read_csv(self.path + e_file)
        #     self.dict_label = {}
        #     for index, row in csv_file.iterrows():
        #         image_id = row['image']
        #         rank = int(row['DR grade'])
        #         self.imgs.append([self.path+self.file+image_id, rank, image_id])       
       
   

        if ddr_pseudo:
            if train:
                img_path = '/remote-home/share/18-houjunlin-18110240004/DR-public-dataset/DDRdataset/DDR-dataset/DR_grading/DDR_preprocess1024/'
                file_path = '/remote-home/share/18-houjunlin-18110240004/MMAC_semi/ddr_psuedolabel/'
                img_list = [{} for _ in range(5)]
                e_files = ['ddr_train_effb1.csv', 'ddr_valid_effb1.csv', 'ddr_test_effb1.csv']
                for e_file in e_files:
                    mode = 'preprocess1024_' + e_file.split('_')[1]
                    csv_file = pd.read_csv(file_path + 'results/' + e_file)
                    for index, row in csv_file.iterrows():
                        image_id = mode + '/' + row['image']
                        rank = int(row['class'])
                        prob = row['prob']
                        img_list[rank][image_id] = prob

                img_list_sorted = []
                for i in range(5):
                    img_list_class = list(zip(img_list[i].values(),img_list[i].keys()))
                    img_list_class_sorted = sorted(img_list_class,reverse=True)
                    img_list_sorted.append(img_list_class_sorted)

                # r = 0.1 # iter 0 
                num_4 = int(ddr_pseudo_r*len(img_list_sorted[4]))
                class_distribution = [8.75, 9, 5, 1.25, 1]
                num = np.array(class_distribution) * num_4
                print(num)
                img_list_selected = []

                for i in range(5):
                    num_ = int(num[i]) if int(num[i])<=len(img_list_sorted[i]) else len(img_list_sorted[i])
                    img_list_selected.append(img_list_sorted[i][:num_])
                
                for i in range(5):
                    num_ = int(num[i]) if int(num[i])<=len(img_list_sorted[i]) else len(img_list_sorted[i])
                    for index in range(num_):
                        image = img_list_selected[i][index][1]
                        self.imgs.append([img_path+image,i,image])



        data_aug = {
            'brightness': 0.2,  # how much to jitter brightness # 0.8,1.2
            'contrast': 0.2,  # How much to jitter contrast
            'saturation': 0.2,
            'hue': 0.05,
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'degrees': (-180, 180),  # range of degrees to select from # vit:-180
            'translate': (0.2, 0.2),  # tuple of maximum absolute fraction for horizontal and vertical translations
            'img_size': 512
        }
        if train:
            self.transform = T.Compose([

                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                    saturation=data_aug['saturation'],
                    hue=data_aug['hue']
                ),
                T.RandomResizedCrop(
                    size=((data_aug['img_size'], data_aug['img_size'])),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),
                T.RandomAffine(
                    degrees=data_aug['degrees'],
                    translate=data_aug['translate']
                ),
                T.RandomGrayscale(0.2),
                T.GaussianBlur(kernel_size=3),
                T.RandomAdjustSharpness(sharpness_factor=2),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])

           
        elif val or test or all:
            self.transform = T.Compose([
                T.Resize((data_aug['img_size'],data_aug['img_size'])),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])

        print(len(self.imgs))
        
    def __getitem__(self, index):

        img, label, name = self.imgs[index]
        data = Image.open(img).convert('RGB')

        if self.train and self.supcon:
            data1 = self.transform(data)
            data2 = self.transform(data)
            return [data1, data2], label, name

        data = self.transform(data)

        if self.lanet:
            label_clf = 0 if label == 0 else 1
            return data, label_clf, label, name

        return data, label, name

    def __len__(self):
        return len(self.imgs)


