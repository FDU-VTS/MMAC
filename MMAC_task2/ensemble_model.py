import argparse
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu

from torchvision.models.segmentation import deeplabv3_resnet50

class model:
    def __init__(self):
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")
        self.model_LC = []
        self.model_CNV = []
        self.model_FS = []

        self.preprocessing = get_preprocessing()

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        for checkpoint_LC in os.listdir(os.path.join(dir_path, 'LC')):
            checkpoint_path_LC = os.path.join(dir_path, 'LC', checkpoint_LC)
            model = torch.load(checkpoint_path_LC, map_location=self.device)
            model.to(self.device)
            model.eval()
            self.model_LC.append(model)


        for checkpoint_CNV in os.listdir(os.path.join(dir_path, 'CNV')):
            checkpoint_path_CNV = os.path.join(dir_path, 'CNV', checkpoint_CNV)
            model = torch.load(checkpoint_path_CNV, map_location=self.device)
            model.to(self.device)
            model.eval()
            self.model_CNV.append(model)


        for checkpoint_FS in os.listdir(os.path.join(dir_path, 'FS')):
            checkpoint_path_FS = os.path.join(dir_path, 'FS', checkpoint_FS)
            model = torch.load(checkpoint_path_FS, map_location=self.device)
            model.to(self.device)
            model.eval()
            self.model_FS.append(model)

    def predict(self, input_image, lesion_type, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param lesion_type: a string indicates the lesion type of the input image: 'Lacquer_Cracks' or 'Choroidal_Neovascularization' or 'Fuchs_Spot'.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: a ndarray indicates the predicted segmentation mask with the shape 800 x 800.
        The pixel value for the lesion area is 255, and the background pixel value is 0.
        """
        image = cv2.resize(input_image, (512, 512))

        # image = input_image / 255
        # image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        image = torch.from_numpy(self.preprocessing(image=image)['image']).unsqueeze(0)

        image = image.to(self.device, torch.float)
        pred_mask = None
        with torch.no_grad():
            if lesion_type == 'Lacquer_Cracks':
                pred_mask = []
                for model in self.model_LC:                
                    pred = model(image)
                    if type(pred) == tuple:
                        pred_mask.append(pred[0])
                    else:
                        pred_mask.append(pred)
                pred_mask = sum(pred_mask)/len(pred_mask)
                pred_mask = pred_mask.detach().squeeze().numpy()
            elif lesion_type == 'Choroidal_Neovascularization':
                pred_mask = []
                for model in self.model_CNV:
                    pred = model(image)
                    if type(pred) == tuple:
                        pred_mask.append(pred[0])
                    else:
                        pred_mask.append(pred)
                new_pred_mask = (pred_mask[0].detach().squeeze().numpy() > 0.5)
                for i in range(1,len(pred_mask)):
                    new_pred_mask = new_pred_mask | (pred_mask[i].detach().squeeze().numpy() > 0.5)
                pred_mask = new_pred_mask 
            elif lesion_type == 'Fuchs_Spot':
                pred_mask = []
                for model in self.model_FS:
                    pred = model(image)
                    if type(pred) == tuple:
                        pred_mask.append(pred[0])
                    else:
                        pred_mask.append(pred)
                pred_mask = sum(pred_mask)/len(pred_mask)
                pred_mask = pred_mask.detach().squeeze().numpy()
        pred_mask = np.array(pred_mask > 0.5, dtype=np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (800, 800), interpolation=cv2.INTER_NEAREST)
        return pred_mask


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing():
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnext50_32x4d','imagenet')

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)
  
