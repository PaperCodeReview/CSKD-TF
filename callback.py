import os
import yaml
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler

from common import create_stamp


def create_callbacks(args, logger, initial_epoch):
    if not args.resume:
        if args.checkpoint or args.history or args.tensorboard:
            if os.path.isdir(f'{args.result_path}/{args.dataset}/{args.stamp}'):
                flag = input(f'\n{args.dataset}/{args.stamp} is already saved. '
                              'Do you want new stamp? (y/n) ')
                if flag == 'y':
                    args.stamp = create_stamp()
                    initial_epoch = 0
                    logger.info(f'New stamp {args.stamp} will be created.')
                elif flag == 'n':
                    return -1, initial_epoch
                else:
                    logger.info(f'You must select \'y\' or \'n\'.')
                    return -2, initial_epoch

            os.makedirs(f'{args.result_path}/{args.dataset}/{args.stamp}')
            yaml.dump(
                vars(args), 
                open(f'{args.result_path}/{args.dataset}/{args.stamp}/model_desc.yml', 'w'), 
                default_flow_style=False)
        else:
            logger.info(f'{args.stamp} is not created due to '
                        f'checkpoint - {args.checkpoint} | '
                        f'history - {args.history} | '
                        f'tensorboard - {args.tensorboard}')

    callbacks = []
    if args.checkpoint:
        os.makedirs(f'{args.result_path}/{args.dataset}/{args.stamp}/checkpoint', exist_ok=True)
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(
                f'{args.result_path}/{args.dataset}/{args.stamp}/checkpoint',
                '{epoch:04d}_{val_loss:.4f}_{val_acc1:.4f}_{val_acc5:.4f}.h5'),
            monitor='val_acc1',
            mode='max',
            verbose=1,
            save_weights_only=True))

    if args.history:
        os.makedirs(f'{args.result_path}/{args.dataset}/{args.stamp}/history', exist_ok=True)
        callbacks.append(CSVLogger(
            filename=f'{args.result_path}/{args.dataset}/{args.stamp}/history/epoch.csv',
            separator=',', append=True))

    if args.tensorboard:
        callbacks.append(TensorBoard(
            log_dir=f'{args.result_path}/{args.dataset}/{args.stamp}/logs',
            histogram_freq=args.tb_histogram,
            write_graph=True, 
            write_images=True,
            update_freq=args.tb_interval,
            profile_batch=2,))

    if args.lr_scheduler:
        def scheduler(epoch):
            if epoch < 100:
                return 0.1
            elif epoch < 150:
                return 0.01
            else:
                return 0.001
                
        callbacks.append(LearningRateScheduler(
            schedule=scheduler,
            verbose=1))

    return callbacks, initial_epoch