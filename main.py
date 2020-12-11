import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from common import set_seed
from common import get_arguments
from common import get_logger
from common import get_session
from common import search_same
from common import create_stamp
from dataloader import set_dataset
from dataloader import DataLoader
from models import set_model
from callback import create_callbacks

import tensorflow as tf


class_dict = {
    'imagenet': 1000,
    'cifar100': 100,
    'tinyimagenet': 200,
    'cub': 200,
}

def main():
    args = get_arguments()
    set_seed(args.seed)
    args.classes = class_dict[args.dataset]
    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        args.stamp = create_stamp()

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info(f"{k} : {v}")


    ##########################
    # Strategy
    ##########################
    if len(args.gpus.split(',')) > 1:
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % num_workers == 0

    logger.info(f"{strategy.__class__.__name__} : {num_workers}")
    logger.info(f"GLOBAL BATCH SIZE : {args.batch_size}")


    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.dataset, args.classes, args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    validation_steps = len(valset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== TRAINSET ==========")
    logger.info(f"    --> {len(trainset)}")
    logger.info(f"    --> {steps_per_epoch}")

    logger.info("=========== VALSET ===========")
    logger.info(f"    --> {len(valset)}")
    logger.info(f"    --> {validation_steps}")


    ##########################
    # Model
    ##########################
    with strategy.scope():
        model = set_model(args.backbone, args.dataset, args.classes)
        if args.snapshot:
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
            temperature=args.temperature,
            run_eagerly=True)


    ##########################
    # Generator
    ##########################
    train_generator = DataLoader(
        loss=args.loss,
        mode='train', 
        datalist=trainset, 
        dataset=args.dataset, 
        classes=args.classes,
        batch_size=args.batch_size, 
        shuffle=True).dataloader()

    val_generator = DataLoader(
        loss='crossentropy',
        mode='val', 
        datalist=valset, 
        dataset=args.dataset, 
        classes=args.classes,
        batch_size=args.batch_size, 
        shuffle=False).dataloader()


    ##########################
    # Train
    ##########################
    callbacks = create_callbacks(args)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,)


if __name__ == '__main__':
    main()