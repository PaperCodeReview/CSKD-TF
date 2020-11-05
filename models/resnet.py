import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


# def regularizer(flag, decay=1e-4):
#     if flag:
#         return l2(decay)
#     return None

def block0(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    if conv_shortcut is True:
        shortcut = Conv2D(filters, 1, strides=stride, 
                          kernel_regularizer=l2(1e-4), 
                          name=name+'_0_conv')(x)
        shortcut = BatchNormalization(name=name+'_0_norm')(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, kernel_size, strides=stride, padding='same', 
               kernel_regularizer=l2(1e-4), name=name+'_1_conv')(x)
    x = BatchNormalization(name=name+'_1_norm')(x)
    x = Activation('relu', name=name+'_1_acti')(x)
    
    x = Conv2D(filters, kernel_size, padding='same', 
               kernel_regularizer=l2(1e-4), name=name+'_2_conv')(x)
    x = BatchNormalization(name=name+'_2_norm')(x)

    x = Add(name=name+'_add')([shortcut, x])
    x = Activation('relu', name=name+'_2_acti')(x)
    return x


def stack0(x, filters, blocks, stride1=2, name=None):
    x = block0(x, filters, stride=stride1, name=name+'_block1')
    for i in range(2, blocks + 1):
        x = block0(x, filters, conv_shortcut=False, name=name+'_block' + str(i))
    return x


def block0_1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    preact = BatchNormalization(name=name+'_pre_norm')(x)
    preact = Activation('relu', name=name+'_pre_acti')(preact)
    if conv_shortcut is True:
        shortcut = Conv2D(filters, 1, strides=stride, 
                          kernel_regularizer=l2(1e-4), name=name+'_0_conv')(preact)
        shortcut = BatchNormalization(name=name+'_0_norm')(shortcut)
    else:
        shortcut = preact

    x = Conv2D(filters, kernel_size, strides=stride, padding='same', 
               kernel_regularizer=l2(1e-4), name=name+'_1_conv')(preact)
    x = BatchNormalization(name=name+'_1_norm')(x)
    x = Activation('relu', name=name+'_1_acti')(x)
    
    x = Conv2D(filters, kernel_size, padding='same', 
               kernel_regularizer=l2(1e-4), name=name+'_2_conv')(x)
    x = Add(name=name+'_add')([shortcut, x])
    return x


def stack0_1(x, filters, blocks, stride1=2, name=None):
    x = block0_1(x, filters, stride=stride1, name=name+'_block1')
    for i in range(2, blocks + 1):
        x = block0_1(x, filters, conv_shortcut=False, name=name+'_block' + str(i))
    return x


def ResNet(
    backbone,
    input_shape,
    classes, 
    stack_fn, 
    preact, 
    use_bias, 
    **kwargs):

    img_input = x = Input(shape=input_shape, name='main_input')
    
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')(x)
    x = Conv2D(64, 3, strides=1, use_bias=use_bias, 
               kernel_regularizer=l2(1e-4), name='conv1_conv')(x)
    if not preact:
        x = BatchNormalization(name='conv1_norm')(x)
        x = Activation('relu', name='conv1_acti')(x)

    x = stack_fn(x)
    if preact:
        x = BatchNormalization(name='post_norm')(x)
        x = Activation('relu', name='post_acti')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, name='main_output')(x)
    model = Model(img_input, x, name=backbone)
    return model


def ResNet18(
    backbone,
    input_shape,
    classes,
    **kwargs):

    def stack_fn(x):
        x = stack0(x, 64, 2, stride1=1, name='conv2')
        x = stack0(x, 128, 2, name='conv3')
        x = stack0(x, 256, 2, name='conv4')
        x = stack0(x, 512, 2, name='conv5')
        return x
    return ResNet(
        backbone=backbone,
        input_shape=input_shape,
        classes=classes, 
        stack_fn=stack_fn, 
        preact=False, 
        use_bias=True, 
        **kwargs)


def PreAct_ResNet18(
    backbone,
    input_shape,
    classes,
    **kwargs):

    def stack_fn(x):
        x = stack0_1(x, 64, 2, stride1=1, name='conv2')
        x = stack0_1(x, 128, 2, name='conv3')
        x = stack0_1(x, 256, 2, name='conv4')
        x = stack0_1(x, 512, 2, name='conv5')
        return x
    return ResNet(
        backbone=backbone,
        input_shape=input_shape,
        classes=classes, 
        stack_fn=stack_fn, 
        preact=True, 
        use_bias=True, 
        **kwargs)