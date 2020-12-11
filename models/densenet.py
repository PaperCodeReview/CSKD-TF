import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

from models.cskd import CSKD


def conv_block(x, growth_rate):
    shortcut = x
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4 * growth_rate, 1, use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, 3, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = Concatenate()([x, shortcut])
    return x


def dense_block(x, growth_rate, blocks):
    for _ in range(blocks):
        x = conv_block(x, growth_rate)
    return x


def transition_block(x, reduction):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), 
               1, use_bias=False, kernel_regularizer=l2(1e-4))
    x = AveragePooling2D(pool_size=2, strides=2)(x)
    return x


def CIFAR_DenseNet(
    input_shape,
    classes,
    blocks,
    growth_rate=12,
    reduction=.5):

    img_input = x = Input(shape=input_shape)

    x = Conv2D(2 * growth_rate, 3, padding='same', use_bias=False)(x)

    x = dense_block(x, growth_rate, blocks[0])
    x = transition_block(x, reduction)
    x = dense_block(x, growth_rate, blocks[1])
    x = transition_block(x, reduction)
    x = dense_block(x, growth_rate, blocks[2])
    x = transition_block(x, reduction)
    x = dense_block(x, growth_rate, blocks[3])
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    logits = Dense(classes, use_bias=True)(x)

    model = CSKD(img_input, logits)
    return model


def CIFAR_DenseNet121(input_shape, classes):
    return CIFAR_DenseNet(input_shape, classes, [6, 12, 24, 16],
                          growth_rate=12, reduction=.5)