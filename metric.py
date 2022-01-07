import tensorflow as tf
from tqdm import tqdm


class CalcMetric:
    def __init__(self, save_npz=True):
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.loss_history = []
        self.metric_history = []
        self.save_npz = save_npz

    def update(self, x, y, model, loss):
        self.epoch_loss_avg.update_state(loss)
        self.epoch_accuracy.update_state(y, model(x, training=False))

    def output_and_reset(self):
        v1 = self.epoch_loss_avg.result().numpy()
        v2 = self.epoch_accuracy.result().numpy()
        self.loss_history.append(v1)
        self.metric_history.append(v2)
        self.reset()
        return v1, v2

    def eval_generator(self, val_data_generator, model, loss_func):
        for i in tqdm(range(len(val_data_generator))):
            x, y = val_data_generator[i]
            loss = loss_func(model, x, y, training=False)
            self.update(x, y, model, loss)

    def reset(self):
        print("Metric reset")
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()