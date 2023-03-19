import tensorflow as tf

class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, units, name="residal", activation=None):
        super().__init__(name=name)

        self.dense_1 = tf.keras.layers.Dense(units, kernel_initializer="glorot_uniform", bias_initializer="zeros" ,activation=activation)
        self.dense_2 = tf.keras.layers.Dense(units, kernel_initializer="glorot_uniform", bias_initializer="zeros", activation=activation)

    def call(self, inputs):
        return inputs + self.dense_2(self.dense_1(inputs))