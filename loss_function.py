import tensorflow as tf

#CE_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def sparse_cross_entropy_loss(y_true, y_pred, num_class=2):
    y_true = tf.one_hot(y_true, depth=num_class)
    softmax_output = tf.exp(y_pred) / tf.reduce_sum(tf.exp(y_pred), axis=1)[:, None]
    y_log_p = tf.math.multiply(y_true, tf.math.log(softmax_output + 1e-8))
    loss = -tf.reduce_mean(y_log_p)
    return loss


def classification_loss(model, x, y, training=True, **kwargs):
    y_pred = model(x, training=training)
    return sparse_cross_entropy_loss(y_true=y, y_pred=y_pred, **kwargs)


def gradient(model, loss, inputs, targets, training=True):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=training)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

