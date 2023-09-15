import albumentations as albu
import torch

def get_training_augmentation():
    train_transform = [
        albu.Resize(height=512,width=512),
        albu.HorizontalFlip(p=0.5),

        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.ShiftScaleRotate(scale_limit=0, rotate_limit=(-180,180), shift_limit=0, p=1, border_mode=0),
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=(-90,90), shift_limit=0.1, p=1, border_mode=0),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.RandomBrightness(limit=0.4),
        albu.RandomContrast(limit=0.4),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),


        # albu.Resize(height=512,width=512),
        # albu.Flip(),
        # albu.ShiftScaleRotate(shift_limit=0.2, rotate_limit=90), # default = A.ShiftScaleRotate()
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(p=1),
        #     albu.RandomGamma(p=1),
        # ]),
        # ##albu.CoarseDropout(max_height=5, min_height=1, max_width=512, min_width=51, mask_fill_value=0),
        # albu.OneOf([
        #     albu.Sharpen(p=1),
        #     albu.Blur(blur_limit=3, p=1),
        #     albu.Downscale(scale_min=0.7, scale_max=0.9, p=1),
        # ]),
        # #A.RandomResizedCrop(512, 512, p=0.2),
        # albu.GridDistortion(p=0.2),
        # albu.CoarseDropout(max_height=128, min_height=32, max_width=128, min_width=32, max_holes=3, p=0.2, mask_fill_value=0.),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
        albu.Resize(height=512,width=512),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return torch.tensor(x.transpose(2, 0, 1).astype('float32'))


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)