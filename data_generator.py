import numpy as np
import tensorflow as tf
from PIL import Image


class ImgDataGen(tf.keras.utils.Sequence):

    def __init__(self, img_path, img_label, batch_size, preprocess_input=None,
                 transform=None, shuffle=True, data_type=np.float32):

        self.img_path = img_path
        self.img_label = img_label
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.index_list = np.arange(len(self.img_path))
        self.data_type = data_type
        self.preprocess_input = preprocess_input

    def __getitem__(self, index):
        idx = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]
        selected_path = self.img_path[idx]
        y = self.img_label[idx]
        img_placeholder = []
        for i_path in selected_path:
            img = np.array(Image.open(i_path))
            if self.transform is not None:
                img = self.transform(image=img)['image']

            img_placeholder.append(img[None, :, :, :])

        x = np.concatenate(img_placeholder).astype(self.data_type)
        if self.preprocess_input is not None:
            x = self.preprocess_input(x)

        return x, y

    def __len__(self):
        return len(self.img_path) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_list)