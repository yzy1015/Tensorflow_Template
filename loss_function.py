import tensorflow as tf

CE_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def classification_loss(model, x, y, training=True):
    y_pred = model(x, training=training)
    return CE_loss_object(y_true=y, y_pred=y_pred)


def gradient(model, loss, inputs, targets, training=True):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=training)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

