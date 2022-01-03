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
learning_rate = 0.1
model = FineTuneNet(efficientnet_base, 2)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
# load data
train_path, train_label, val_path, val_label = cat_dog_dataset_train_val(config.train_dir, config.num_val_img)
train_data_generator = ImgDataGen(train_path, train_label, config.batch_size,
                                  preprocess_input=preprocess_input,
                                  transform=config.img_transform)

train_data_generator.on_epoch_end() # whether to shuffle the data at beginning
val_data_generator = ImgDataGen(val_path, val_label, config.val_batch_size,
                                preprocess_input=preprocess_input,
                                transform=config.img_transform)

current_best_loss = None
current_best_acc = None
val_loss_history = []
train_loss_history = []
for epoch in range(config.EP_NUM):
    print("Epoch:", epoch)
    # define metric
    epoch_train_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # train for 1 epoch
    #for i in tqdm(range(len(train_data_generator))): //todo
    for i in tqdm(range(5)):
        X, y = train_data_generator[i]
        loss, grad_vector = gradient(model, classification_loss, X, y)
        optimizer.apply_gradients(zip(grad_vector, model.trainable_variables))
        epoch_train_loss_avg.update_state(loss)
        epoch_train_accuracy.update_state(y, model(X, training=False))

    train_data_generator.on_epoch_end()
    train_loss = epoch_train_loss_avg.result().numpy()
    train_acc = epoch_train_accuracy.result().numpy()
    train_loss_history.append(train_loss)
    print('Train loss:', train_loss,
          'Train accuracy:', train_acc)

    # evaluate on val data
    for i in tqdm(range(len(val_data_generator))):
        X, y = val_data_generator[i]
        epoch_val_loss_avg.update_state(classification_loss(model, X, y, training=False))
        epoch_val_accuracy.update_state(y, model(X, training=False))

    val_loss = epoch_val_loss_avg.result().numpy()
    val_acc = epoch_val_accuracy.result().numpy()
    val_loss_history.append(val_loss)
    print('Val loss:', val_loss,
          'Val accuracy:', val_acc)

    # loss based lr decay
    if (len(train_loss_history) > 1) and (train_loss_history[-2] * config.lr_decay_threshold < train_loss):
        learning_rate = learning_rate / config.lr_decay_ratio
        optimizer.lr.assign(learning_rate)
        print('learning rate decay to', optimizer.lr.read_value().numpy())

    if (current_best_loss is None) or (val_acc > current_best_acc) or (val_loss < current_best_loss):
        current_best_acc = val_acc
        current_best_loss = val_loss
        model.save(config.save_format.format(epoch, val_loss, val_acc))

    # optimizer.lr.assign(0.001)

    ''' restart if using adam optimizer
    if epoch % config.restart_adam_num == 0:
        for var in optimizer.variables():
            var.assign(tf.zeros_like(var))
    '''

