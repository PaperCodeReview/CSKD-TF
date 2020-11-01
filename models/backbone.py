import tensorflow as tf
from models.resnet import ResNet18
from models.resnet import PreAct_ResNet18
# from models.densenet import DenseNet


def set_model(
    model: str,
    dataset: str,
    classes: int):

    if dataset in ['cifar100', 'tinyimagenet']:
        # preact resnet-18, densenet-bc
        if model == 'resnet18':
            backbone = PreAct_ResNet18(
                backbone=model,
                dataset=dataset,
                input_shape=(32, 32, 3),
                classes=classes)
        elif model == 'densenetbc':
            raise NotImplementedError('TODO')
        else:
            raise ValueError('the models of cs-kd for cifar100 or tinyimagenet '
                             'were only resnet18 and densenet-bc.')
    
    elif dataset in ['imagenet']:
        # resnet-50, resnet-101, resnext-101-32x4d
        if model == 'resnet50':
            backbone = tf.keras.applications.ResNet50(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        elif model == 'resnet101':
            backbone = tf.keras.applications.ResNet101(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        elif model == 'resnext101':
            raise NotImplementedError()

    else:
        # resnet-18, densenet-121
        if 'resnet' in model:
            backbone = ResNet18(
                backbone=model,
                dataset=dataset,
                input_shape=(224, 224, 3),
                classes=classes)
        elif 'densenet' in model:
            backbone = tf.keras.applications.DenseNet121(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        else:
            raise ValueError()

    return backbone