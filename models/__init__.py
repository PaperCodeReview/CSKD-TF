import tensorflow as tf
from .resnet import ResNet18
from .resnet import PreAct_ResNet18
from .resnet import ResNet50
from .resnet import ResNet101
from .densenet import CIFAR_DenseNet121


def set_model(
    name: str,
    dataset: str,
    classes: int):
    """
    Args
        name (str) : backbone's name
        dataset (str) : dataset which will be used
        classes (int) : the number of classes

    Returns
        backbone (tensorflow.keras.Model) : backbone network which will be trained
    """
    if dataset in ['cifar100', 'tinyimagenet']:
        # preact resnet-18, densenet-bc
        if name == 'resnet18':
            backbone = PreAct_ResNet18(
                backbone=name,
                input_shape=(32, 32, 3),
                classes=classes)
        elif name == 'densenetbc':
            raise NotImplementedError('densenetbc is not implemented yet.')
            # backbone = CIFAR_DenseNet121(
            #     input_shape=(32, 32, 3),
            #     classes=classes)
        else:
            raise ValueError('the models of cs-kd for cifar100 or tinyimagenet '
                             'were only resnet18 and densenet-bc.')
    
    elif dataset in ['imagenet']:
        # resnet-50, resnet-101, resnext-101-32x4d
        if name == 'resnet50':
            backbone = ResNet50(
                backbone=name,
                input_shape=(224, 224, 3),
                classes=classes)
        elif name == 'resnet101':
            backbone = ResNet101(
                backbone=name,
                input_shape=(224, 224, 3),
                classes=classes)
        elif name == 'resnext101':
            raise NotImplementedError()

    else:
        # resnet-18, densenet-121
        if 'resnet' in name:
            backbone = ResNet18(
                backbone=name,
                input_shape=(224, 224, 3),
                classes=classes)
        elif 'densenet' in name:
            raise NotImplementedError('densenet is not implemented yet.')
            # backbone = tf.keras.applications.DenseNet121(
            #     include_top=True,
            #     weights=None,
            #     input_shape=(224, 224, 3),
            #     classes=classes)
        else:
            raise ValueError()

    return backbone