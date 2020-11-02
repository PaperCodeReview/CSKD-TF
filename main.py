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
from models.backbone import set_model
from models.backbone import CSKD
from callback import OptionalLearningRateSchedule
from callback import create_callbacks

import tensorflow as tf


class_dict = {
    'imagenet': 1000,
    'cifar100': 100,
    'tinyimagenet': 200,
}

def main():
    set_seed()
    args = get_arguments()
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
        logger.info("{} : {}".format(k, v))


    ##########################
    # Strategy
    ##########################
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    # num_workers = strategy.num_replicas_in_sync
    # assert args.batch_size % num_workers == 0

    # logger.info('{} : {}'.format(strategy.__class__.__name__, num_workers))
    # logger.info("GLOBAL BATCH SIZE : {}".format(args.batch_size))


    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.dataset, args.classes, args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    validation_steps = len(valset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== TRAINSET ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== VALSET ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))


    ##########################
    # Model & Metric & Generator
    ##########################
    lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
    model = set_model(args.backbone, args.dataset, args.classes)
    if args.loss in ['cls', 'consistency']:
        train_generator = DataLoader(
            mode='train', 
            datalist=trainset, 
            dataset=args.dataset, 
            classes=args.classes,
            batch_size=args.batch_size, 
            shuffle=True).cskd_dataloader()
        
        val_generator = DataLoader(
            mode='val', 
            datalist=valset, 
            dataset=args.dataset, 
            classes=args.classes,
            batch_size=args.batch_size, 
            shuffle=False).xe_dataloader()

        cskd = CSKD(model)
        cskd.compile(
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            cls_lambda=args.loss_weight,
            temperature=args.temperature)

        cskd.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.batch_size,
            # callbacks=
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,)
    else:
        train_generator = DataLoader(
            mode='train', 
            datalist=trainset, 
            dataset=args.dataset, 
            classes=args.classes,
            batch_size=args.batch_size, 
            shuffle=True).xe_dataloader()
        
        val_generator = DataLoader(
            mode='val', 
            datalist=valset, 
            dataset=args.dataset, 
            classes=args.classes,
            batch_size=args.batch_size, 
            shuffle=False).xe_dataloader()

        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['acc'])

        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.batch_size,
            # callbacks=
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,)



    
    # metrics = {
    #     'loss_xe'       : tf.keras.metrics.Mean('loss_xe', dtype=tf.float32),
    #     'acc1'          : tf.keras.metrics.TopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
    #     'acc5'          : tf.keras.metrics.TopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32),
    #     'val_loss_xe'   : tf.keras.metrics.Mean('val_loss_xe', dtype=tf.float32),
    #     'val_acc1'      : tf.keras.metrics.TopKCategoricalAccuracy(1, 'val_acc1', dtype=tf.float32),
    #     'val_acc5'      : tf.keras.metrics.TopKCategoricalAccuracy(5, 'val_acc5', dtype=tf.float32),}
    # if args.loss in ['cls', 'consistency']:
    #     metrics.update({'loss_cls'     : tf.keras.metrics.Mean('loss_cls', dtype=tf.float32),
    #                     'val_loss_cls' : tf.keras.metrics.Mean('val_loss_cls', dtype=tf.float32),})
    #     if args.loss == 'consistency':
    #         metrics.update({'loss_consistency'      : tf.keras.metrics.Mean('loss_consistency', dtype=tf.float32),
    #                         'val_loss_consistency'  : tf.keras.metrics.Mean('val_loss_consistency', dtype=tf.float32),})

    # # with strategy.scope():
    # model = set_model(args.backbone, args.dataset, args.classes)
    # if args.summary:
    #     model.summary()
    #     return
    
    # # optimizer & loss
    # lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
    # optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001)
    # criterion_xe = tf.keras.losses.categorical_crossentropy
    # if args.loss in ['cls', 'consistency']:
    #     criterion_cls = tf.keras.losses.KLD
    #     if args.loss == 'consistency':
    #         criterion_consistency = tf.keras.losses.KLD

    # # generator
    # if args.loss == 'crossentropy':
    #     train_generator = DataLoader(
    #         mode='train', 
    #         datalist=trainset, 
    #         dataset=args.dataset, 
    #         classes=args.classes,
    #         batch_size=args.batch_size, 
    #         shuffle=True).xe_dataloader()
    #     # train_generator = strategy.experimental_distribute_dataset(train_generator)
    #     train_iterator = iter(train_generator)
    # elif args.loss == 'cls':
    #     train_iterator = DataLoader(
    #         mode='train', 
    #         datalist=trainset, 
    #         dataset=args.dataset, 
    #         classes=args.classes,
    #         batch_size=args.batch_size, 
    #         shuffle=True).cskd_dataloader()
    #     # train_generator = strategy.experimental_distribute_values_from_function(train_generator)
    # elif args.loss == 'consistency':
    #     raise NotImplementedError()

    # cnt = 0
    # for t in train_iterator:
    #     # print(t[0]['main_input'].shape, tf.reduce_min(t[0]['main_input']), tf.reduce_max(t[0]['main_input']), t[1]['main_output'])
    #     print(cnt, t[0]['img1'].shape, t[0]['img2'].shape, t[1]['main_output'].shape)
    #     cnt += 1

    # val_generator = DataLoader(
    #     mode='val', 
    #     datalist=valset, 
    #     dataset=args.dataset, 
    #     classes=args.classes,
    #     batch_size=args.batch_size, 
    #     shuffle=False).xe_dataloader()
    # # val_generator = strategy.experimental_distribute_dataset(val_generator)
    # val_iterator = iter(val_generator)

    # csvlogger, train_writer, val_writer = create_callbacks(args, metrics)
    # logger.info("Build Model & Metrics")

    
    # ##########################
    # # READY Train
    # ##########################
    # def do_step(iterator, mode):
    #     def get_loss(inputs, labels):
    #         loss = 0.
    #         if args.loss == 'crossentropy':
    #             logits = tf.cast(model(inputs['main_input']), tf.float32)
    #             loss_xe = tf.nn.compute_average_loss(
    #                 criterion_xe(labels['main_output'], logits, from_logits=True),
    #                 global_batch_size=args.batch_size)
    #             loss += loss_xe
    #             loss_dict = {'loss': loss, 'loss_xe': loss_xe}
    #             return loss_dict, logits
    #         elif args.loss == 'cls':
    #             img1, img2 = inputs['img1'], inputs['img2']
    #             logits1 = tf.cast(model(img1), tf.float32)
    #             logits2 = tf.cast(model(tf.stop_gradient(img2)), tf.float32)
    #             loss_xe = tf.nn.compute_average_loss(
    #                 criterion_xe(labels['main_output'], logits1, from_logits=True),
    #                 global_batch_size=args.batch_size)
    #             loss_cls = tf.nn.compute_average_loss(
    #                 criterion_cls(
    #                     tf.nn.softmax(logits1/args.temperature),
    #                     tf.nn.softmax(logits2/args.temperature)),
    #                 global_batch_size=args.batch_size)
    #             loss += loss_xe + args.loss_weight * (args.temperature ** 2) * loss_cls
    #             loss_dict = {'loss': loss, 'loss_xe': loss_xe, 'loss_cls': loss_cls}
    #             return loss_dict, logits1
    #         elif args.loss == 'consistency':
    #             raise NotImplementedError()

    #     def step_fn(from_iter):
    #         inputs, labels = from_iter
    #         if mode == 'train':
    #             with tf.GradientTape() as tape:
    #                 loss_dict, logits = get_loss(inputs, labels)

    #             grads = tape.gradient(loss_dict['loss'], model.trainable_variables)
    #             optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    #         else:
    #             loss_dict, logits = get_loss(inputs, labels)
            
    #         metrics['acc1' if mode == 'train' else 'val_acc1'].update_state(labels['main_output'], logits)
    #         metrics['acc5' if mode == 'train' else 'val_acc5'].update_state(labels['main_output'], logits)
    #         return loss_dict
            
    #     # loss_dict_per_replica = strategy.run(step_fn, args=(next(iterator),))
    #     # if num_workers > 1:
    #     #     loss_dict = {k: tf.reduce_sum(v.values) for k, v in loss_dict.items()}
    #     # else:
    #     #     loss_dict = loss_dict_per_replica
    #     loss_dict = step_fn(next(iterator))
    #     for k, v in loss_dict.items():
    #         if k == 'loss':
    #             continue
    #         metrics[k if mode == 'train' else 'val_'+k].update_state(v)
        

    # ##########################
    # # Train
    # ##########################
    # for epoch in range(initial_epoch, args.epochs):
    #     print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
    #     print('Learning Rate : {}'.format(optimizer.learning_rate(optimizer.iterations)))

    #     # train
    #     print('Train')
    #     progBar_train = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=metrics.keys())
    #     for step in range(steps_per_epoch):
    #         do_step(train_iterator, 'train')
    #         progBar_train.update(step, values=[(k, v.result()) for k, v in metrics.items() if not 'val' in k])

    #         if args.tensorboard and args.tb_interval > 0:
    #             if (epoch*steps_per_epoch+step) % args.tb_interval == 0:
    #                 with train_writer.as_default():
    #                     for k, v in metrics.items():
    #                         if not 'val' in k:
    #                             tf.summary.scalar(k, v.result(), step=epoch*steps_per_epoch+step)

    #     if args.tensorboard and args.tb_interval == 0:
    #         with train_writer.as_default():
    #             for k, v in metrics.items():
    #                 if not 'val' in k:
    #                     tf.summary.scalar(k, v.result(), step=epoch)

    #     # val
    #     print('\n\nValidation')
    #     progBar_val = tf.keras.utils.Progbar(validation_steps, stateful_metrics=metrics.keys())
    #     for step in range(validation_steps):
    #         do_step(val_iterator, 'val')
    #         progBar_val.update(step, values=[(k, v.result()) for k, v in metrics.items() if 'val' in k])
    
    #     # logs
    #     logs = {k: v.result().numpy() for k, v in metrics.items()}
    #     logs['epoch'] = epoch + 1

    #     print()
    #     if args.checkpoint:
    #         ckpt_path = '{:04d}_{:.4f}_{:.4f}_{:.4f}.h5'.format(epoch+1, logs['val_loss'], logs['val_acc1'], logs['val_acc5'])
    #         model.save_weights(
    #             f'{args.result_path}/{args.dataset}/{args.stamp}/checkpoint/{ckpt_path}')

    #         print('Saved at {}'.format(
    #             f'{args.result_path}/{args.dataset}/{args.stamp}/checkpoint/{ckpt_path}'))

    #     if args.history:
    #         csvlogger = csvlogger.append(logs, ignore_index=True)
    #         csvlogger.to_csv('{args.result_path}/{args.dataset}/{args.stamp}/history/epoch.csv', index=False)

    #     if args.tensorboard:
    #         with train_writer.as_default():
    #             tf.summary.scalar('loss', metrics['loss'].result(), step=epoch)
    #             tf.summary.scalar('acc1', metrics['acc1'].result(), step=epoch)
    #             tf.summary.scalar('acc5', metrics['acc5'].result(), step=epoch)

    #         with val_writer.as_default():
    #             tf.summary.scalar('val_loss', metrics['val_loss'].result(), step=epoch)
    #             tf.summary.scalar('val_acc1', metrics['val_acc1'].result(), step=epoch)
    #             tf.summary.scalar('val_acc5', metrics['val_acc5'].result(), step=epoch)
        
    #     for k, v in metrics.items():
    #         v.reset_states()



if __name__ == '__main__':
    main()