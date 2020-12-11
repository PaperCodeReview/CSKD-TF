import os
import sys
import yaml
import random
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",       type=str,       default='resnet50')
    parser.add_argument("--dataset",        type=str,       default='imagenet',
                        choices=['cifar100', 'tinyimagenet', 'imagenet', 'cub', 'stanforddogs', 'mit67'])
                        
    parser.add_argument("--batch_size",     type=int,       default=128)
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=200)

    parser.add_argument("--lr",             type=float,     default=.1)
    parser.add_argument("--loss",           type=str,       default='crossentropy',
                        choices=['crossentropy', 'cls', 'consistency'])
    parser.add_argument("--temperature",    type=float,     default=4.,     
                        choices=[1., 4.])
    parser.add_argument("--loss_weight",    type=float,     default=1.,
                        choices=[1., 2., 3., 4.])

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--lr_scheduler",   action='store_true')
    parser.add_argument("--tb_interval",    type=int,       default=0)
    parser.add_argument("--tb_histogram",   type=int,       default=0)

    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument("--resume",         action='store_true')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument('--seed',           type=int,       default=42)
    parser.add_argument("--gpus",           type=str,       default='-1')
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore-search",  type=str,       default='')

    return parser.parse_args()


def set_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger


def get_session(args):
    assert int(tf.__version__.split('.')[0]) >= 2.0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.gpus != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def create_stamp():
    weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    temp = datetime.now()
    return "{:02d}{:02d}{:02d}_{}_{:02d}_{:02d}_{:02d}".format(
        temp.year // 100,
        temp.month,
        temp.day,
        weekday[temp.weekday()],
        temp.hour,
        temp.minute,
        temp.second,
    )


def search_same(args):
    search_ignore = ['checkpoint', 'history', 'snapshot', 'summary',
                     'src_path', 'data_path', 'result_path', 
                     'epochs', 'stamp', 'gpus', 'ignore_search']
    if len(args.ignore_search) > 0:
        search_ignore += args.ignore_search.split(',')

    initial_epoch = 0
    stamps = os.listdir(f'{args.result_path}/{args.dataset}')
    for stamp in stamps:
        try:
            desc = yaml.full_load(
                open(f'{args.result_path}/{args.dataset}/{stamp}/model_desc.yml', 'r'))
        except:
            continue
        
        flag = True
        save_flag = False
        for k, v in vars(args).items():
            if k in search_ignore:
                continue

            if k == 'resume' and k not in desc:
                desc[k] = False
                save_flag = True
                
            if v != desc[k]:
                # if stamp == '201104_Wed_08_53_35':
                print(stamp, k, desc[k], v)
                flag = False
                break

        if save_flag:
            yaml.dump(
                desc, 
                open(f'{args.result_path}/{args.dataset}/{stamp}/model_desc.yml', 'w'), 
                default_flow_style=False)
            save_flag = False
        
        if flag:
            args.stamp = stamp
            try:
                df = pd.read_csv(
                    f'{args.result_path}/{args.dataset}/{args.stamp}/history/epoch.csv')
            except:
                raise ValueError('history loading error!')
            
            if len(df) > 0:
                if int(df['epoch'].values[-1]+1) == args.epochs:
                    print(f'{stamp} Training already finished!!!')
                    return args, -1

                elif np.isnan(df['val_loss'].values[-1]) or np.isinf(df['val_loss'].values[-1]):
                    print('{} | Epoch {:04d}: Invalid loss, terminating training'.format(stamp, int(df['epoch'].values[-1]+1)))
                    return args, -1

                else:
                    ckpt_list = sorted(
                        [d for d in os.listdir(
                            f'{args.result_path}/{args.dataset}/{args.stamp}/checkpoint') if 'h5' in d],
                        key=lambda x: int(x.split('_')[0]))
                    
                    if len(ckpt_list) > 0:
                        args.snapshot = f'{args.result_path}/{args.dataset}/{args.stamp}/checkpoint/{ckpt_list[-1]}'
                        initial_epoch = int(ckpt_list[-1].split('_')[0])
                    else:
                        print('{} Training already finished!!!'.format(stamp))
                        return args, -1
            else:
                continue

            break
    
    return args, initial_epoch