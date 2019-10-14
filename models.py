import os
import torch
import segmentation_models_pytorch as smp

# 'densenet201', 'se_resnext101_32x4d', 'resnet18'
#ENCODER = 'resnet50'
#ENCODER = 'efficientnet-b2'
#ENCODER = 'efficientnet-b4'

#ENCODER = 'densenet201'

ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

ACTIVATION = None
#preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def create_model(encoder_type, ckp=None):
    model = smp.Unet(
        encoder_name=encoder_type, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=4, 
        activation=ACTIVATION,
    )
    #model_file = './logs/segmentation/checkpoints/best.pth'
    if ckp and os.path.exists(ckp):
        print('loading {}...'.format(ckp))
        model.load_state_dict(torch.load(ckp)['model_state_dict'])

    return model


def test_model():
    model = create_model('efficientnet-b2', ckp='./logs_eb2_lb652/checkpoints/best.pth').cuda()
    x = torch.randn(2, 3, 320, 640).cuda()
    y = model(x)
    print(y.size())

if __name__ == '__main__':
    test_model()