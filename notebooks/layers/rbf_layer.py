import tensorflow as tf
from .basis_functions import *

class RbfLayer(tf.keras.layers.Layer):
    def __init__(self, N_rbf, c, name="rbf", **kwargs):
        super().__init__(name=name, **kwargs)

        self.N_rbf = N_rbf
        self.c = c

        def freq_init(n, dtype):
            return tf.constant(1/c * np.pi * np.arange(1, n + 1), dtype=dtype)
        
        self.frequencies = self.add_weight(name="frequencies", shape=self.N_rbf,
                                           initializer=freq_init, trainable=True)

    def call(self, inputs):
        d = inputs
        d = tf.expand_dims(d, -1)

        return e_rbf(self.frequencies * d, self.c)
