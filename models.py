import os
import torch
import segmentation_models_pytorch as smp

# 'densenet201'
#ENCODER = 'resnet50'
#ENCODER = 'efficientnet-b2'
#ENCODER = 'efficientnet-b4'

ENCODER = 'densenet201'

ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

ACTIVATION = None
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def create_model():
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=4, 
        activation=ACTIVATION,
    )
    #model_file = './logs/segmentation/checkpoints/best.pth'
    #if os.path.exists(model_file):
    #    print('loading {}...'.format(model_file))
    #    model.load_state_dict(torch.load(model_file))

    return model
