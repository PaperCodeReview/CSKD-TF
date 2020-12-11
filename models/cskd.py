import tensorflow as tf


class CSKD(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(CSKD, self).__init__(*args, **kwargs)

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
            xe_logits = self(img, training=True)
            xe_loss = self.xe_loss(label, xe_logits, from_logits=True)
            xe_loss = tf.reduce_sum(xe_loss) / self.batch_size
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(xe_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(label, xe_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': xe_loss})
        return results

    def train_cskd(self, data):
        img1, label1, img2, _ = data
        cls_logits = self(img2, training=False)
        with tf.GradientTape() as tape:
            xe_logits = self(img1, training=True)
            xe_loss = self.xe_loss(label1, xe_logits, from_logits=True)
            xe_loss = tf.reduce_sum(xe_loss) / self.batch_size

            cls_loss = self.cls_loss(
                tf.nn.softmax(tf.stop_gradient(cls_logits) / self.temperature),
                tf.nn.softmax(xe_logits / self.temperature))
            cls_loss = tf.reduce_sum(cls_loss) * (self.temperature**2) / self.batch_size

            loss = xe_loss + self.cls_lambda * cls_loss

        trainable_vars = self.trainable_variables
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
        xe_logits = self(img, training=False)
        xe_loss = self.xe_loss(label, xe_logits, from_logits=True)
        xe_loss = tf.reduce_sum(xe_loss) / self.batch_size

        self.compiled_metrics.update_state(label, xe_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': xe_loss})
        return results