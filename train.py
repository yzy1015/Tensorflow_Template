import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import config
from data_generator import ImgDataGen
from model import FineTuneNet
from loss_function import classification_loss, gradient
from load_file_name import cat_dog_dataset_train_val


preprocess_input = tf.keras.applications.efficientnet.preprocess_input
efficientnet_base = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False,
                                                                      weights='imagenet',
                                                                      input_shape=(224, 224, 3))
efficientnet_base.trainable = False
model = FineTuneNet(efficientnet_base, 2)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# load data
train_path, train_label, val_path, val_label = cat_dog_dataset_train_val(config.train_dir, config.num_val_img)
train_data_generator = ImgDataGen(train_path, train_label, config.batch_size,
                                preprocess_input=preprocess_input,
                                transform=config.img_transform)

val_data_generator = ImgDataGen(val_path, val_label, config.val_batch_size,
                                preprocess_input=preprocess_input,
                                transform=config.img_transform)

'''
for epoch in range(config.EP_NUM):
    print("Epoch:", epoch)
    # define metric
    epoch_train_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # train for 1 epoch
    for i in tqdm(range(len(train_data_generator))):
        X, y = train_data_generator[i]
        loss, grad_vector = gradient(model, classification_loss, X, y)
        optimizer.apply_gradients(zip(grad_vector, model.trainable_variables))
        epoch_train_loss_avg.update_state(loss)
        epoch_train_accuracy.update_state(y, model(X, training=False))

    train_data_generator.on_epoch_end()

    print('Train loss:', epoch_train_loss_avg.result().numpy(),
          'Train accuracy:', epoch_train_accuracy.result().numpy())

    # evaluate on val data
    for i in tqdm(range(len(val_data_generator))):
        X, y = val_data_generator[i]
        epoch_val_loss_avg.update_state(classification_loss(model, X, y, training=False))
        epoch_val_accuracy.update_state(y, model(X, training=False))

    print('Val loss:', epoch_val_loss_avg.result().numpy(),
          'Val accuracy:', epoch_val_accuracy.result().numpy())

'''

for epoch in range(8):
    print("Epoch:", epoch)
    # define metric
    epoch_train_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # train for 1 epoch
    train_data_generator.on_epoch_end()
    for i in tqdm(range(5)):
        X, y = train_data_generator[i]
        loss, grad_vector = gradient(model, classification_loss, X, y)
        optimizer.apply_gradients(zip(grad_vector, model.trainable_variables))
        epoch_train_loss_avg.update_state(loss)
        epoch_train_accuracy.update_state(y, model(X, training=False))

    train_data_generator.on_epoch_end()

    print('Train loss:', epoch_train_loss_avg.result().numpy(),
          'Train accuracy:', epoch_train_accuracy.result().numpy())

    # evaluate on val data
    for i in tqdm(range(len(val_data_generator))):
        X, y = val_data_generator[i]
        epoch_val_loss_avg.update_state(classification_loss(model, X, y, training=False))
        epoch_val_accuracy.update_state(y, model(X, training=False))

    print('Val loss:', epoch_val_loss_avg.result().numpy(),
          'Val accuracy:', epoch_val_accuracy.result().numpy())