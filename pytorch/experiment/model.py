import torchvision.models as models


def load_model(model_name, pretrained):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f'Unknown model {model_name}')

    return model
