import tensorflow as tf
from .basis_functions import *
from .basis_functions import a_sbf

class SbfLayer(tf.keras.layers.Layer):
    def __init__(self, N_shbf, N_srbf, c, name="sbf", **kwargs):
        super().__init__(name=name, **kwargs)

        self.N_shbf = N_shbf
        self.N_srbf = N_srbf
        self.c = c

    def call(self, inputs):
        d, alpha = inputs
        return np.array([tf.numpy_function(a_sbf, [l, n, d, alpha, self.c], tf.float32) for l in range(self.N_shbf) for n in range(1, self.N_srbf + 1)])
