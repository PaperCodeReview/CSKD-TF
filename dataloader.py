import os
import cv2
import tqdm
import random
import numpy as np
import pandas as pd
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(dataset, classes=None, data_path=None):
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        trainset = [[i, l] for i, l in zip(x_train, y_train)]
        valset = [[i, l] for i, l in zip(x_test, y_test)]
    else:
        trainset = pd.read_csv(f'{data_path}/{dataset}_trainset.csv').values.tolist()
        trainset = [[f'{data_path}/{t[0]}', t[1]] for t in trainset]
        valset = pd.read_csv(f'{data_path}/{dataset}_valset.csv').values.tolist()
        valset = [[f'{data_path}/{v[0]}', v[1]] for v in valset]
    
    return np.array(trainset, dtype='object'), np.array(valset, dtype='object')


class DataLoader:
    def __init__(self, loss, mode, datalist, dataset, classes, batch_size=128, shuffle=True):
        self.loss = loss
        self.mode = mode
        self.datalist = datalist
        self.dataset = dataset
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.augment = Augment(self.dataset)

    def __len__(self):
        return len(self.datalist)

    def fetch_dataset(self, path, y):
        x = tf.io.read_file(path)
        return tf.data.Dataset.from_tensors((x, y))

    def augmentation(self, img, label, shape):
        if self.dataset == 'tinyimagenet':
            if self.mode == 'train':
                img = self.augment._crop(img, shape)
                img = self.augment._random_hflip(img)

            img = self.augment._resize(img, 32)

        elif self.dataset.startswith('cifar'):
            if self.mode == 'train':
                img = self.augment._pad(img, [[4, 4], [4, 4], [0, 0]])
                img = self.augment._cifar_crop(img)
                img = self.augment._random_hflip(img)

        else:
            if self.mode == 'train':
                img = self.augment._crop(img, shape)
                img = self.augment._resize(img, 224)
                img = self.augment._random_hflip(img)

            else:
                img = self.augment._resize(img, 256)
                img = self.augment._center_crop(img, 224/256)
            
        img = self.augment._standardize(img)

        # one-hot encoding
        label = tf.squeeze(tf.one_hot(label, self.classes))
        return (img, label)

    def xe_dataloader(self):
        def _preprocess_image(img, label):
            if self.dataset in ['cifar100']:
                shape = (32, 32, 3)
            else:
                shape = tf.image.extract_jpeg_shape(img)
                img = tf.io.decode_jpeg(img, channels=3)
            return self.augmentation(img, label, shape)

        imglist, labellist = self.datalist[:,0].tolist(), self.datalist[:,1].tolist()
        dataset = tf.data.Dataset.from_tensor_slices((imglist, labellist))
        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(self.datalist))

        if 'cifar' not in self.dataset:
            dataset = dataset.interleave(self.fetch_dataset, num_parallel_calls=AUTO)

        dataset = dataset.map(_preprocess_image, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def cskd_dataloader(self):
        # set numpy generator -> tf.data.Dataset.from_generator
        def _imgload(img):
            if self.dataset in ['cifar100']:
                return img
            return cv2.imread(img)[...,::-1]
            
        def _loader():
            imglist, labellist = self.datalist[:,0], self.datalist[:,1].astype(np.int)
            indices = np.arange(len(self.datalist))
            while True:
                if self.shuffle:
                    indices = np.random.permutation(len(self.datalist))
                for idx in indices:
                    img1, label1 = _imgload(imglist[idx]), labellist[idx]
                    idx_cls = np.random.choice(np.where(labellist == label1)[0])
                    img2, label2 = _imgload(imglist[idx_cls]), labellist[idx_cls]
                    assert label1 == label2, 'label and label2 must be equal!'
                    yield (img1, label1, img1.shape, img2, label2, img2.shape)

        def _preprocess_image(img1, label1, shape1, img2, label2, shape2):
            img1, label1 = self.augmentation(img1, label1, shape1)
            img2, label2 = self.augmentation(img2, label2, shape2)
            return (img1, label1, img2, label2)

        dataset = tf.data.Dataset.from_generator(
            _loader,
            output_types=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
            output_shapes=(
                tf.TensorShape([None, None, None,]), tf.TensorShape([]), tf.TensorShape([None,]),
                tf.TensorShape([None, None, None,]), tf.TensorShape([]), tf.TensorShape([None,])))
        dataset = dataset.map(_preprocess_image, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def dataloader(self):
        if self.loss == 'crossentropy':
            return self.xe_dataloader()
        else:
            return self.cskd_dataloader()
        

class Augment:
    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.mean_std = [(0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)]
        else:
            self.mean_std = [(0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225)]

    def _standardize(self, x):
        x = tf.cast(x, tf.float32)
        x /= 255.
        x -= self.mean_std[0]
        x /= self.mean_std[1]
        return x

    def _pad(self, x, paddings):
        return tf.pad(x, paddings)

    def _cifar_crop(self, x):
        offset_height = tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int32)
        offset_width = tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int32)
        x = tf.slice(x, [offset_height, offset_width, 0], [32, 32, 3])
        return x
        
    def _crop(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            aspect_ratio_range=(3/4, 4/3),
            area_range=(.08, 1.),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)
        
        offset_height, offset_width, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        x = tf.slice(x, [offset_height, offset_width, 0], [target_height, target_width, 3])
        return x

    def _center_crop(self, x, central_fraction):
        return tf.image.central_crop(x, central_fraction)

    def _resize(self, x, img_size):
        return tf.image.resize(x, (img_size, img_size))

    def _random_hflip(self, x):
        return tf.image.random_flip_left_right(x)