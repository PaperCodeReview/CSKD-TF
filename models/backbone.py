import tensorflow as tf
from models.resnet import ResNet18
from models.resnet import PreAct_ResNet18
from models.densenet import CIFAR_DenseNet121


def set_model(
    name: str,
    dataset: str,
    classes: int):
    """
    Args
        name (str) : backbone's name
        dataset (str) : dataset which will be used
        classes (int) : the number of classes

    Returns
        backbone (tensorflow.keras.Model) : backbone network which will be trained
    """
    if dataset in ['cifar100', 'tinyimagenet']:
        # preact resnet-18, densenet-bc
        if name == 'resnet18':
            backbone = PreAct_ResNet18(
                backbone=name,
                dataset=dataset,
                input_shape=(32, 32, 3),
                classes=classes)
        elif name == 'densenetbc':
            backbone = CIFAR_DenseNet121(
                input_shape=(32, 32, 3),
                classes=classes)
        else:
            raise ValueError('the models of cs-kd for cifar100 or tinyimagenet '
                             'were only resnet18 and densenet-bc.')
    
    elif dataset in ['imagenet']:
        # resnet-50, resnet-101, resnext-101-32x4d
        if name == 'resnet50':
            backbone = tf.keras.applications.ResNet50(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        elif name == 'resnet101':
            backbone = tf.keras.applications.ResNet101(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=classes)
        elif name == 'resnext101':
            raise NotImplementedError()

    else:
        # resnet-18, densenet-121
        if 'resnet' in name:
            backbone = ResNet18(
                backbone=name,
                dataset=dataset,
                input_shape=(224, 224, 3),
                classes=classes)
        elif 'densenet' in name:
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
        loss,
        batch_size,
        optimizer,
        metrics,
        xe_loss,
        cls_loss,
        cls_lambda,
        temperature,
        run_eagerly=None):

        super(CSKD, self).compile(optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly)
        self.loss = loss
        self.batch_size = batch_size
        self.xe_loss = xe_loss
        self.cls_loss = cls_loss
        self.cls_lambda = cls_lambda
        self.temperature = temperature

    def train_xe(self, data):
        img, label = data
        with tf.GradientTape() as tape:
            xe_logits = self.model(img, training=True)
            xe_loss = self.xe_loss(label, xe_logits, from_logits=True)
            xe_loss = tf.reduce_sum(xe_loss) / self.batch_size
        
        trainable_vars = self.model.trainable_variables
        grads = tape.gradient(xe_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(label, xe_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': xe_loss})
        return results

    def train_cskd(self, data):
        img1, label1, img2, label2 = data
        cls_logits = self.model(img2, training=False)
        with tf.GradientTape() as tape:
            xe_logits = self.model(img1, training=True)
            xe_loss = self.xe_loss(label1, xe_logits, from_logits=True)
            xe_loss = tf.reduce_sum(xe_loss) / self.batch_size

            # TF : loss = y_true * log(y_true / y_pred)
            # torch : loss = y * (log(y) - x)
            cls_loss = self.cls_loss(
                tf.nn.softmax(tf.stop_gradient(cls_logits) / self.temperature),
                tf.nn.softmax(xe_logits / self.temperature))
            cls_loss = tf.reduce_sum(cls_loss) * (self.temperature**2) / self.batch_size

            loss = xe_loss + self.cls_lambda * cls_loss

        trainable_vars = self.model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(label1, xe_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss, 'xe_loss': xe_loss, 'cls_loss': cls_loss})
        return results

    def train_step(self, data):
        if self.loss == 'crossentropy':
            return self.train_xe(data)
        else:
            return self.train_cskd(data)

    def test_step(self, data):
        img, label = data
        xe_logits = self.model(img, training=False)
        xe_loss = self.xe_loss(label, xe_logits, from_logits=True)
        xe_loss = tf.reduce_sum(xe_loss) / self.batch_size

        self.compiled_metrics.update_state(label, xe_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': xe_loss})
        return results