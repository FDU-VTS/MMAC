import torchvision.models as models
import torch
import torch.nn as nn
import sys
from torch import optim
import timm

from biomedgpt_resnet import ResNet
from pmcclip_resnet import ModifiedResNet
# import open_clip

def get_model(name,num_class=5):
    if name == 'resnet50':
        net = models.resnet50(weights="IMAGENET1K_V2")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
    elif name == 'resnet50_ddrprew':
        net = models.resnet50(weights="IMAGENET1K_V2")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
        weight = '/remote-home/share/18-houjunlin-18110240004/MMAC/dr_checkpoints/93.pkl'
        state_dict = torch.load(weight)['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,True)
        net.fc = nn.Linear(in_channel, num_class) 
    elif name == 'resnet50_ddrpre':
        net = models.resnet50(weights="IMAGENET1K_V2")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
        weight = '/remote-home/share/18-houjunlin-18110240004/MMAC/dr_checkpoints/52.pkl'
        state_dict = torch.load(weight)['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)
        net.fc = nn.Linear(in_channel, num_class) 
    elif name == 'resnet50_kagglepre':
        net = models.resnet50(weights="IMAGENET1K_V2")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
        weight = '/remote-home/share/18-houjunlin-18110240004/MMAC/dr_checkpoints/kaggle_res50_ce_79.pkl'
        state_dict = torch.load(weight)['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,True)
        net.fc = nn.Linear(in_channel, num_class) 
    elif name == 'resnet18':
        net = models.resnet18(weights="IMAGENET1K_V1")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
    elif name == 'resnet34':
        net = models.resnet34(weights="IMAGENET1K_V1")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
    elif name == 'wide_resnet50_2':
        net = models.wide_resnet50_2(weights="IMAGENET1K_V1")
        print('change clsifier')
    elif name == 'googlenet':
        net = models.googlenet(weights="IMAGENET1K_V1")
        print('change clsifier')
    elif name == 'vgg16': 
        net = models.vgg16_bn(weights="IMAGENET1K_V1")
        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, num_class)
    elif name == 'inceptionv3': 
        net = models.inception_v3(weights="IMAGENET1K_V1")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class)
    elif name == 'mobilev3':
        net = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        net.classifier[3] = nn.Linear(1280, num_class)
    elif name == 'densenet121': 
        net = models.densenet121(weights="IMAGENET1K_V1")
        in_channel = net.classifier.in_features
        # net.classifier = nn.Sequential(
        #     nn.Linear(in_channel, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, num_class)
        # )
        net.classifier = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb0':
        net = models.efficientnet_b0(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb1':
        net = models.efficientnet_b1(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb2':
        net = models.efficientnet_b2(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb3':
        net = models.efficientnet_b3(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class) 
    elif name == 'efficientnetb4':
        net = models.efficientnet_b4(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)  
    elif name == 'efficientnetb5':
        net = models.efficientnet_b5(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb6':
        net = models.efficientnet_b6(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)  
    elif name == 'efficientnetb7':
        net = models.efficientnet_b7(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class) 
    elif name == 'vitb16':
        net = models.vit_b_16(weights="IMAGENET1K_V1")
        in_channel = net.heads[-1].in_features
        net.heads[-1] = nn.Linear(in_channel, num_class)
    elif name == 'vitl16':
        net = models.vit_l_16(weights="IMAGENET1K_V1")
        in_channel = net.heads[-1].in_features
        net.heads[-1] = nn.Linear(in_channel, num_class)
    elif name == 'swint':
        net = models.swin_t(weights='IMAGENET1K_V1')
        in_channel = net.head.in_features
        net.head = nn.Linear(in_channel, num_class)
    elif name == 'swins':
        net = models.swin_s(weights='IMAGENET1K_V1')
        in_channel = net.head.in_features
        net.head = nn.Linear(in_channel, num_class)
    elif name == 'swinb':
        net = models.swin_b(weights='IMAGENET1K_V1')
        in_channel = net.head.in_features
        net.head = nn.Linear(in_channel, num_class)
    elif name == 'uni4eye': #224
        net = timm.create_model('vit_large_patch16_224')
        in_channel = net.head.in_features
        net.head = nn.Linear(in_channel, num_class)
        uni4eye_weight = 'Uni4Eye++_vit_large_224x224_e99.pth'
        net.load_state_dict(torch.load(uni4eye_weight)['model'],strict=False)
    elif name == 'ssit': #224
        from vits import vit_small_patch16_384
        # net = timm.create_model('vit_small_patch16_224')
        net = vit_small_patch16_384()
        # in_channel = net.head.in_features
        # net.head = nn.Linear(in_channel, num_class)
        ssit_weight = 'pretrained_vits_imagenet_initialized.pt'
        net.load_state_dict(torch.load(ssit_weight)['state_dict'],strict=True)
    elif name == 'convnext_t':
        net = models.convnext_tiny(weights='IMAGENET1K_V1')
        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, num_class)
    elif name == 'convnext_s':
        net = models.convnext_small(weights='IMAGENET1K_V1')
        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, num_class)
    elif name == 'convnext_b':
        net = models.convnext_base(weights='IMAGENET1K_V1')
        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, num_class)
    elif name == 'convnext_l':
        net = models.convnext_large(weights='IMAGENET1K_V1')
        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, num_class)
    elif name == 'squeezenet1_0':
        net = models.squeezenet1_0(weights='IMAGENET1K_V1')
        in_channel = net.classifier[1].in_channels
        net.classifier[1] = nn.Conv2d(in_channel, num_class, kernel_size=(1, 1), stride=(1, 1))

    elif name == 'seresnext50':
        net = timm.create_model('seresnext50_32x4d',pretrained=True, num_classes=num_class)
    elif name == 'seresnext101':
        net = timm.create_model('seresnext101_32x8d',pretrained=True, num_classes=num_class)
    elif name == 'biomedgpt_tiny': #256
        net = ResNet([3, 4, 6])
        weight = 'biomedgpt_tiny.pt'
        state_dict = torch.load(weight)['model']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("encoder.embed_images.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)
    elif name == 'biomedgpt_medium': #256
        net = ResNet([3, 8, 36])
        weight = 'biomedgpt_medium.pt'
        state_dict = torch.load(weight)['model']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("encoder.embed_images.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)
    elif name == 'pmcclip512':
        net = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=512)
        weight = 'checkpoint.pt'
        state_dict = torch.load(weight)['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.visual.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)
    elif name == 'pmcclip224':
        net = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224)
        weight = 'checkpoint.pt'
        state_dict = torch.load(weight)['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.visual.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)
    elif name == 'lesion_based_resnet50': # 512
        net = models.resnet50(weights="IMAGENET1K_V2")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class)
        weight = torch.load('/remote-home/share/18-houjunlin-18110240004/DRmethod/Lesion-based-Contrastive-Learning-main/lesion_based_CL_trained_weights/resnet50_128_08.pt' )
        net.load_state_dict(weight, strict=False)
    elif name == 'pubmedclip': # 512
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        net = model.visual
        net.head = nn.Linear(768, num_class)
    elif name == 'xm_rotation_resnet18': # 224
        net = models.resnet18(weights="IMAGENET1K_V1")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 128)
        weight = '/remote-home/share/18-houjunlin-18110240004/MMAC/savedmodels/DR-pretrain-model.pth.tar'
        state_dict = torch.load(weight)['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)
        net.fc = nn.Linear(in_channel, num_class)

    elif name == 'xm_ssl_resnet18': # 224
        net = models.resnet18(weights="IMAGENET1K_V1")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 128)
        weight = '/remote-home/share/18-houjunlin-18110240004/MMAC/savedmodels/fold0-epoch-2000.pth.tar'
        state_dict = torch.load(weight)['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,True)
        net.fc = nn.Linear(in_channel, num_class)
    else:
        print('no model found')
        sys.exit(0)
    return net

def get_lr_scheduler(name, optimizer):
    if name == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    elif name == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-9)
    else:
        sys.exit('lr_scheduler not found')
    return lr_scheduler



if __name__=='__main__':
    from thop import profile
    net = get_model('efficientnetb1',num_class=5)
    input_tensor = torch.randn(1,3,128,128)
    flops, params = profile(net, inputs=(input_tensor,))

    from thop import clever_format
    flops, params = clever_format([flops, params], "%.3f")


    print("FLOPs:", flops)
    print("Params:", params)
