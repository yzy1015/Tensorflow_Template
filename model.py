import tensorflow as tf
from tensorflow.keras import regularizers


class FineTuneNet(tf.keras.Model):
    def __init__(self, base_net=None, n_class=2, dense_num1=32):
        super(FineTuneNet, self).__init__()
        self.base_net = base_net
        self.dense1 = tf.keras.layers.Dense(dense_num1, kernel_regularizer=regularizers.l2(0.001))
        self.dense2 = tf.keras.layers.Dense(n_class, kernel_regularizer=regularizers.l2(0.001))

    def call(self, x):
        x = self.base_net(x)
        gap_layer = tf.keras.layers.GlobalAveragePooling2D()
        x = gap_layer(x)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        return x
