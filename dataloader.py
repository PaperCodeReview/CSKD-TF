import os
import tqdm
import random
import numpy as np
import pandas as pd
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(dataset, data_path):
    if 'cifar' in  dataset:
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        b_trainset = unpickle(os.path.join(data_path, 'train'))
        b_valset = unpickle(os.path.join(data_path, 'test'))
        
        trainset_temp = {}
        valset_temp = {}
        for i, l in enumerate(b_trainset[b'fine_labels']):
            if not l in trainset_temp.keys():
                temp = np.dstack((b_trainset[b'data'][i][:1024].reshape((32,32)),
                                b_trainset[b'data'][i][1024:2048].reshape((32,32)),
                                b_trainset[b'data'][i][2048:].reshape((32,32))))
                trainset_temp[l] = temp[np.newaxis,...]
            else:
                temp = np.dstack((b_trainset[b'data'][i][:1024].reshape((32,32)),
                                b_trainset[b'data'][i][1024:2048].reshape((32,32)),
                                b_trainset[b'data'][i][2048:].reshape((32,32))))
                trainset_temp[l] = np.append(trainset_temp[l], temp[np.newaxis,...], axis=0)
                
        for i, l in enumerate(b_valset[b'fine_labels']):
            if not l in valset_temp.keys():
                temp = np.dstack((b_valset[b'data'][i][:1024].reshape((32,32)),
                                b_valset[b'data'][i][1024:2048].reshape((32,32)),
                                b_valset[b'data'][i][2048:].reshape((32,32))))
                valset_temp[l] = temp[np.newaxis,...]
            else:
                temp = np.dstack((b_valset[b'data'][i][:1024].reshape((32,32)),
                                b_valset[b'data'][i][1024:2048].reshape((32,32)),
                                b_valset[b'data'][i][2048:].reshape((32,32))))
                valset_temp[l] = np.append(valset_temp[l], temp[np.newaxis,...], axis=0)
                
        trainset = []
        valset = []
        for k, v in trainset_temp.items():
            for vv in v:
                trainset.append([vv, k])
                
        for k, v in valset_temp.items():
            for vv in v:
                valset.append([vv, k])
    else:
        trainset = pd.read_csv(
            os.path.join(
                data_path, '{}_trainset.csv'.format(dataset)
            )).values.tolist()
        valset = pd.read_csv(
            os.path.join(
                data_path, '{}_valset.csv'.format(dataset)
            )).values.tolist()

    return np.array(trainset, dtype='object'), np.array(valset, dtype='object')


class DataLoader:
    def __init__(self, mode, datalist, dataset, classes, batch_size=128, shuffle=True):
        self.mode = mode,
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
            img = self.augment._resize(img, (32, 32, 3))

        elif self.dataset.startswith('cifar'):
            if self.mode == 'train':
                img = self.augment._pad(img, [[4, 4], [4, 4]])
                img = self.augment._crop(img, (40, 40, 3))
                img = self.augment._resize(img, (32, 32, 3))
                img = self.augment._random_hflip(img)

        else:
            if self.mode == 'train':
                img = self.augment._crop(img, shape)
                img = self.augment._resize(img, (224, 224, 3))
                img = self.augment._random_hflip(img)
            else:
                img = self.augment._resize(img, (256, 256, 3))
                img = self.augment._center_crop(img, 224/256)
            
        img = self.augment._standardize(img)

        # one-hot encoding
        label = tf.one_hot(label, self.classes)
        return (img, label)

    def preprocess_image(self, img, label):
        if self.dataset in ['cifar100', 'tinyimagenet']:
            shape = (32, 32, 3)
        else:
            shape = tf.image.extract_jpeg_shape(img)
            img = tf.io.decode_jpeg(img, channels=3)
        img, label = self.augmentation(img, label, shape)
        return ({'main_input': img}, {'main_output': label})

    def xe_dataloader(self):
        imglist, labellist = self.datalist[:,0].tolist(), self.datalist[:,1].tolist()
        dataset = tf.data.Dataset.from_tensor_slices((imglist, labellist))
        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(self.datalist))

        if 'cifar' not in self.dataset:
            dataset = dataset.interleave(self.fetch_dataset, num_parallel_calls=AUTO)
        dataset = dataset.map(self.preprocess_image, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def cskd_dataloader(self):
        # set numpy generator -> tf.data.Dataset.from_generator
        pass
        


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
        
    def _crop(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            aspect_ratio_range=(3/4, 4/3),
            area_range=(.08, 1.),
            max_attempts=100,
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