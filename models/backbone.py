import tensorflow as tf
from models.resnet import ResNet18
from models.resnet import PreAct_ResNet18
# from models.densenet import DenseNet


def set_model(
    model: str,
    dataset: str,
    classes: int):

    if dataset in ['cifar100', 'tinyimagenet']:
        # preact resnet-18, densenet-bc
        if model == 'resnet18':
            backbone = PreAct_ResNet18(
                backbone=model,
                dataset=dataset,
                input_shape=(32, 32, 3),
                classes=classes)
        elif model == 'densenetbc':
            raise NotImplementedError('TODO')
        else:
            raise ValueError('the models of cs-kd for cifar100 or tinyimagenet '
                             'were only resnet18 and densenet-bc.')
    
    elif dataset in ['imagenet']:
        # resnet-50, resnet-101, resnext-101-32x4d
        if model == 'resnet50':
            backbone = tf.keras.applications.ResNet50(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        elif model == 'resnet101':
            backbone = tf.keras.applications.ResNet101(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        elif model == 'resnext101':
            raise NotImplementedError()

    else:
        # resnet-18, densenet-121
        if 'resnet' in model:
            backbone = ResNet18(
                backbone=model,
                dataset=dataset,
                input_shape=(224, 224, 3),
                classes=classes)
        elif 'densenet' in model:
            backbone = tf.keras.applications.DenseNet121(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        else:
            raise ValueError()

    return backbone


class CSKD(tf.keras.Model):
    def __init__(self, model):
        super(CSKD, self).__init__()
        self.model = model

    def compile(
        self,
        optimizer,
        metrics,
        cls_lambda,
        temperature):
        super(CSKD, self).compile(optimizer=optimizer, metrics=metrics)
        self.xe_loss = tf.keras.losses.categorical_crossentropy
        self.cls_loss = tf.keras.losses.KLD
        self.cls_lambda = cls_lambda
        self.temperature = temperature

    def train_step(self, data):
        imgs, labels = data
        img1, img2 = tf.split(imgs, num_or_size_splits=2, axis=1)
        img1, img2 = tf.squeeze(img1), tf.squeeze(img2)
        label1, label2 = tf.split(labels, num_or_size_splits=2, axis=1)
        label1, label2 = tf.squeeze(label1), tf.squeeze(label2)

        cls_logits = self.model(tf.stop_gradient(img2), training=False)
        with tf.GradientTape() as tape:
            xe_logits = self.model(img1, training=True)
            xe_loss = self.xe_loss(label1, xe_logits, from_logits=True)
            xe_loss = tf.reduce_mean(xe_loss)
            cls_loss = self.cls_loss(
                tf.nn.softmax(cls_logits / self.temperature),
                tf.nn.softmax(xe_logits / self.temperature))
            cls_loss = tf.reduce_mean(cls_loss)
            loss = xe_loss + self.cls_lambda * (self.temperature**2) * cls_loss

        trainable_vars = self.model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(label1, xe_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'xe_loss': xe_loss, 'cls_loss': cls_loss})
        return results

    def test_step(self, data):
        img, label = data
        xe_logits = self.model(img, training=False)
        xe_loss = self.xe_loss(label, xe_logits, from_logits=True)
        xe_loss = tf.reduce_mean(xe_loss)

        self.compiled_metrics.update_state(label, xe_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'xe_loss': xe_loss})
        return results
