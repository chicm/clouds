import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from unet_model import create_unet_model
# 'densenet201', 'se_resnext101_32x4d', 'resnet18'
#ENCODER = 'resnet50'
#ENCODER = 'efficientnet-b2'
#ENCODER = 'efficientnet-b4'

#ENCODER = 'densenet201'

ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

ACTIVATION = None
#preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def create_model(encoder_type, ckp=None, act=None):
    global ENCODER_WEIGHTS
    if encoder_type in ['resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d']:
        ENCODER_WEIGHTS = 'instagram'
    if encoder_type.startswith('myunet'):
        nlayers = int(encoder_type.split('_')[1])
        model = create_unet_model(nlayers)
    else:
        model = smp.Unet(
            encoder_name=encoder_type, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=4, 
            activation=act,
        )
    #model_file = './logs/segmentation/checkpoints/best.pth'
    if ckp and os.path.exists(ckp):
        print('loading {}...'.format(ckp))
        model.load_state_dict(torch.load(ckp)['model_state_dict'])

    return model

class UNetC(smp.Unet):
    def __init__(self, encoder_type, encoder_weights='imagenet', classes=4, activation=None):
        super().__init__(encoder_name=encoder_type, encoder_weights=encoder_weights, classes=classes, activation=activation)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.encoder.out_shapes[0], classes)

    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, 0.2, self.training)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        x = self.encoder(x)
        cls_logits = self.logits(x[0])
        x = self.decoder(x)
        return x, cls_logits


def test_model():
    model = create_model('efficientnet-b2', ckp='./logs_eb2_lb652/checkpoints/best.pth').cuda()
    x = torch.randn(2, 3, 320, 640).cuda()
    y = model(x)
    print(y.size())

def test_unetc():
    model = UNetC('densenet161').cuda()
    x = torch.randn(2, 3, 320, 640).cuda()
    y = model(x)
    print(y[0].size())


if __name__ == '__main__':
    #test_model()
    test_unetc()
