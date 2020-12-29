import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

from models.cskd import CSKD


def transition_block(x, reduction, name):
    x = BatchNormalization(epsilon=1.001e-5,
                           name=name+'_bn')(x)
    x = Activation('relu', name=name+'_relu')(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), 1,
               use_bias=False,
               kernel_regularizer=l2(1e-4),
               name=name+'_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name+'_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    x1 = BatchNormalization(epsilon=1.001e-5,
                            name=name+'_0_bn')(x)
    x1 = Activation('relu', name=name+'_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                use_bias=False,
                kernel_regularizer=l2(1e-4),
                name=name+'_1_conv')(x1)
    x1 = BatchNormalization(epsilon=1.001e-5,
                            name=name+'_1_bn')(x1)
    x1 = Activation('relu', name=name+'_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(1e-4),
                name=name+'_2_conv')(x1)
    x = Concatenate(axis=-1, name=name+'_concat')([x, x1])
    return x


def dense_block(x, growth_rate, blocks, name):
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name+'_block'+str(i+1))
    return x


def DenseNet(
    backbone,
    input_shape,
    classes,
    blocks,
    growth_rate=32,
    reduction=.5):

    img_input = x = Input(shape=input_shape)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
    x = Conv2D(64, 7, strides=2, use_bias=False, 
               kernel_regularizer=l2(1e-4),
               name='conv1/conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, growth_rate, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, growth_rate, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, growth_rate, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, growth_rate, blocks[3], name='conv5')

    x = BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, use_bias=False, 
              kernel_regularizer=l2(1e-4),
              name='fc')(x)
    model = CSKD(img_input, x, name=backbone)
    return model


def DenseNet121(backbone, input_shape, classes):
    return DenseNet(backbone, input_shape, classes, [6, 12, 24, 16],
                    growth_rate=32, reduction=.5)
                    

def CIFAR_DenseNet(
    backbone,
    input_shape,
    classes,
    blocks,
    growth_rate=12,
    reduction=.5):

    img_input = x = Input(shape=input_shape)

    x = Conv2D(2 * growth_rate, 3, padding='same', use_bias=False)(x)

    x = dense_block(x, growth_rate, blocks[0], name='conv2')
    x = transition_block(x, reduction, name='pool2')
    x = dense_block(x, growth_rate, blocks[1], name='conv3')
    x = transition_block(x, reduction, name='pool3')
    x = dense_block(x, growth_rate, blocks[2], name='conv4')
    x = transition_block(x, reduction, name='pool4')
    x = dense_block(x, growth_rate, blocks[3], name='conv5')
    
    x = BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, use_bias=False, 
              kernel_regularizer=l2(1e-4),
              name='fc')(x)

    model = CSKD(img_input, x, name=backbone)
    return model


def CIFAR_DenseNet121(backbone, input_shape, classes):
    return CIFAR_DenseNet(backbone, input_shape, classes, [6, 12, 24, 16],
                          growth_rate=12, reduction=.5)