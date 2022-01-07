import tensorflow as tf
from tqdm import tqdm
import config
from data_generator import ImgDataGen
from model import FineTuneNet
from loss_function import gradient
from loss_function import classification_loss as tf_loss_func
from optimizer import update_lr
from load_file_name import cat_dog_dataset_train_val
from metric import CalcMetric


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

# define metric
train_metric_calc = CalcMetric()
val_metric_calc = CalcMetric()
current_best_loss = None
current_best_metric = None
val_loss_history = []
for epoch in range(config.EP_NUM):
    print("Epoch:", epoch)
    # train for 1 epoch
    for i in tqdm(range(len(train_data_generator))):
        X, y = train_data_generator[i]
        loss, grad_vector = gradient(model, tf_loss_func, X, y)
        optimizer.apply_gradients(zip(grad_vector, model.trainable_variables))
        train_metric_calc.update(X, y, model, loss)

    train_data_generator.on_epoch_end() # data shuffling
    train_loss, train_metric = train_metric_calc.output_and_reset()
    print('Train loss:', train_loss, 'Train metric:', train_metric)

    # evaluate on val data
    for i in tqdm(range(len(val_data_generator))):
        X, y = val_data_generator[i]
        loss = tf_loss_func(model, X, y, training=False)
        val_metric_calc.update(X, y, model, loss)

    val_loss, val_metric = val_metric_calc.output_and_reset()
    print('Val loss:', val_loss, 'Val metric:', val_metric)

    # loss based lr decay
    if update_lr(train_metric_calc.loss_history, config, train_loss):
        learning_rate = learning_rate / config.lr_decay_ratio
        optimizer.lr.assign(learning_rate)
        print('learning rate decay to', optimizer.lr.read_value().numpy())

    # if performance on val data better than previous best model, save the new model
    if (current_best_loss is None) or (val_metric > current_best_metric) or (val_loss < current_best_loss):
        current_best_metric = val_metric
        current_best_loss = val_loss
        model.save_weights(config.save_format.format(epoch, val_loss, val_metric))

    # optimizer.lr.assign(0.001)

    ''' restart if using adam optimizer
    if epoch % config.restart_adam_num == 0:
        for var in optimizer.variables():
            var.assign(tf.zeros_like(var))
    '''

