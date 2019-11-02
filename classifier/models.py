import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.model import Model

from segmentation_models_pytorch.encoders import get_encoder

# 'densenet201', 'se_resnext101_32x4d', 'resnet18'
#ENCODER = 'resnet50'
#ENCODER = 'efficientnet-b2'
#ENCODER = 'efficientnet-b4'

#ENCODER = 'densenet201'

ENCODER_WEIGHTS = 'imagenet'

def create_model(encoder_type, work_dir='./work_dir', ckp=None, activation=None):
    model = CloudNet(encoder_type, encoder_weights=ENCODER_WEIGHTS, classes=4, activation=activation)
    if ckp is None:
        ckp = os.path.join(work_dir, encoder_type, 'checkpoints', 'best.pth')
    print('{} exists: {}'.format(ckp, os.path.exists(ckp)))
    if os.path.exists(ckp):
        print('loading {}...'.format(ckp))
        w = torch.load(ckp)
        if 'model_state_dict' in w:
            model.load_state_dict(w['model_state_dict'])
        else:
            model.load_state_dict(w)

    return model, ckp

class CloudNet(Model):
    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=4,
            activation=None,
    ):
        super().__init__()
        self.encoder = get_encoder(encoder_name, encoder_weights=encoder_weights)
        self.name = encoder_name
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.encoder.out_shapes[0], classes)
    
    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, 0.2, self.training)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x, activation=None):
        x = self.encoder(x)
        cls_logits = self.logits(x[0])
        if not self.training:
            if activation is not None:
                cls_logits = activation(cls_logits)
        return cls_logits



class UNetC(smp.Unet):
    def __init__(self, encoder_type, encoder_weights='imagenet', classes=4, activation=None, out_mask=True, out_cls=True):
        super().__init__(encoder_name=encoder_type, encoder_weights=encoder_weights, classes=classes, activation=activation)
        self.activation = activation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.encoder.out_shapes[0], classes)
        self.out_cls = out_cls
        self.out_mask = out_mask
        if not (out_cls or out_mask):
            raise ValueError('out_mask or out_cls')

    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, 0.2, self.training)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        x = self.encoder(x)
        cls_logits = self.logits(x[0])
        x = self.decoder(x)

        if not self.out_mask:
            return cls_logits
        if not self.out_cls:
            return x
        return x, cls_logits


def test_model():
    model = create_model('efficientnet-b2')[0].cuda()
    x = torch.randn(2, 3, 320, 640).cuda()
    y = model(x)
    print(y.size())

def test_unetc():
    model = create_model('densenet161').cuda()
    x = torch.randn(2, 3, 320, 640).cuda()
    y = model(x)
    print(y[0].size())


if __name__ == '__main__':
    test_model()
    #test_unetc()
