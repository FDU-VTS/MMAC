import os
import csv
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class MMACDataset(BaseDataset):
    """MMAC Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            lesion,
            augmentation=None,
            preprocessing=None,
    ):
        if lesion == 'LC':
            images_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/1. Images/1. Training Set'
            masks_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/2. Groundtruths/1. Training Set'
        elif lesion == 'CNV':
            images_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/2. Choroidal Neovascularization/1. Images/1. Training Set'
            masks_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/2. Choroidal Neovascularization/2. Groundtruths/1. Training Set'
        elif lesion == 'FS':
            images_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/3. Fuchs Spot/1. Images/1. Training Set'
            masks_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/3. Fuchs Spot/2. Groundtruths/1. Training Set'
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])

        mask = cv2.imread(self.masks_fps[i], 0)
        mask = (mask == 255).astype('float')[:,:,None]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.augmentation == None and self.preprocessing == None:
            return image, mask, os.path.split(self.images_fps[i])[-1]
        
        return image, mask

    def __len__(self):
        return len(self.ids)


class MMACDataset_task1(BaseDataset):
    """MMAC Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            augmentation=None,
            preprocessing=None,
    ):
        
        # images_dir = './data/1. Classification of Myopic Maculopathy/1. Images/1. Training Set'
        images_dir = '/remote-home/share/21-xiaofan-21210860085/MMAC/unlabel_data/data'
        csv_file = './data/1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv'
        
        self.ids = os.listdir(images_dir)
        # reader = csv.reader(open(csv_file,'r'))
        # header = next(reader)
        # self.ids = []
        # for row in reader:
        #     if int(row[1]) > 0:
        #         self.ids.append(row[0])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        if self.augmentation == None and self.preprocessing == None:
            return image, os.path.split(self.images_fps[i])[-1]
        
        return image

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    # from preprocess import get_training_augmentation

    # train_dataset = MMACDataset(
    #     augmentation=get_training_augmentation(),
    #     lesion='FS',
    # )

    # for i,data in enumerate(train_dataset):
    #     cv2.imwrite("./show/"+str(i)+".png",data[0])
    import segmentation_models_pytorch as smp
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnext50_32x4d', 'imagenet')
    print(preprocessing_fn)
   