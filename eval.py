import os
import yaml
import argparse
import tensorflow as tf

from common import set_seed
from common import get_logger
from common import get_session
from dataloader import set_dataset
from dataloader import DataLoader
from models import set_model


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp",       type=str,       default='resnet50')
    parser.add_argument("--dataset",        type=str,       default='imagenet',
                        choices=['cifar100', 'tinyimagenet', 'imagenet', 'cub', 'stanforddogs', 'mit67'])

    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument("--gpus",           type=str,       default='-1')

    return parser.parse_args()


def main():
    temp_args = get_arguments()
    assert temp_args.snapshot is not None, 'snapshot must be selected!'
    set_seed()

    args = argparse.ArgumentParser().parse_args(args=[])
    tmp = yaml.full_load(
        open(f'{temp_args.result_path}/'
             f'{temp_args.dataset}/'
             f'{temp_args.stamp}/'
              'model_desc.yml', 'r'))

    for k, v in tmp.items():
        setattr(args, k, v)

    args.snapshot = temp_args.snapshot
    args.src_path = temp_args.src_path
    args.data_path = temp_args.data_path
    args.result_path = temp_args.result_path
    args.gpus = temp_args.gpus
    args.batch_size = 1

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info(f"{k} : {v}")


    ##########################
    # Dataset
    ##########################
    _, valset = set_dataset(args.dataset, args.classes, args.data_path)
    validation_steps = len(valset)

    logger.info("TOTAL STEPS OF DATASET FOR EVALUATION")
    logger.info("=========== VALSET ===========")
    logger.info(f"    --> {validation_steps}")


    ##########################
    # Model & Generator
    ##########################
    model = set_model(args.backbone, args.dataset, args.classes)
    model.load_weights(args.snapshot)
    logger.info(f"Load weights at {args.snapshot}")

    model.compile(
        loss=args.loss,
        batch_size=args.batch_size,
        optimizer=tf.keras.optimizers.SGD(args.lr, momentum=.9),
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='acc1'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc5')],
        xe_loss=tf.keras.losses.categorical_crossentropy,
        cls_loss=tf.keras.losses.KLD,
        cls_lambda=args.loss_weight,
        temperature=args.temperature)

    val_generator = DataLoader(
        loss='crossentropy',
        mode='val', 
        datalist=valset, 
        dataset=args.dataset, 
        classes=args.classes,
        batch_size=args.batch_size, 
        shuffle=False).dataloader()

    
    ##########################
    # Evaluation
    ##########################
    print(model.evaluate(val_generator, steps=validation_steps, return_dict=True))


if __name__ == '__main__':
    main()