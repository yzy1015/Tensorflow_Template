import os
import numpy as np


def cat_dog_dataset_train_val(train_dir, num_val_img):
    # train image
    dog_img = np.array([i for i in os.listdir(train_dir) if 'dog' in i])
    cat_img = np.array([i for i in os.listdir(train_dir) if 'cat' in i])
    np.random.shuffle(dog_img)
    np.random.shuffle(cat_img)
    train_img_seq = list(dog_img[:-num_val_img]) + list(cat_img[:-num_val_img])
    val_img_seq = list(dog_img[-num_val_img:]) + list(cat_img[-num_val_img:])
    train_path = [train_dir + i for i in train_img_seq]
    val_path = [train_dir + i for i in val_img_seq]

    train_label = []
    for i in train_path:
        if 'dog' in i:
            train_label = train_label + [0]
        else:
            train_label = train_label + [1]

    val_label = []
    for i in val_path:
        if 'dog' in i:
            val_label = val_label + [0]
        else:
            val_label = val_label + [1]

    train_path = np.array(train_path)
    val_path = np.array(val_path)
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    return train_path, train_label, val_path, val_label
