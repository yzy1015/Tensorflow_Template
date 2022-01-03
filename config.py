import albumentations as A


img_transform = A.Compose([
    A.augmentations.geometric.resize.Resize(288, 288),
    A.augmentations.geometric.transforms.Affine(scale=0.9, translate_px=16,
                                                p=0.5, rotate=(-30, 30), shear=(-5, 5)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.augmentations.transforms.ColorJitter(hue=0.05, p=0.5),
    A.RandomCrop(width=224, height=224)
])

train_dir = '../data/train/'
test_dir = '../data/test/'
num_val_img = 250
EP_NUM = 8
batch_size = 64
val_batch_size = 50
restart_adam_num = 32
lr_decay_threshold = 0.9
lr_decay_ratio = 10
running_loss_plate_length = 3
save_format = './model/epoch_{}_loss_{:.2f}_acc_{:.2f}.h5'