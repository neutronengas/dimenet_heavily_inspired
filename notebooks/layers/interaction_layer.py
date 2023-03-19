import tensorflow as tf
import numpy as np
from .residual_layer import ResidualLayer

class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, n_features, N_bilinear, name="interaction", activation=None):
        super().__init__(name=name)
        self.n_features = n_features
        self.N_bilinear = N_bilinear

        self.dense_rbf = tf.keras.layers.Dense(n_features, use_bias=False)
        self.dense_sbf = tf.keras.layers.Dense(N_bilinear, use_bias=False)

        self.dense_ji = tf.keras.layers.Dense(n_features, activation=activation)
        self.dense_kj = tf.keras.layers.Dense(n_features, activation=activation)

        self.bilin = self.add_weight(name="bilinear", shape=(n_features, self.N_bilinear, self.n_features), 
                                     dtype=np.float32, initializer=tf.initializers.RandomNormal(mean=0.0, stddev=2 / n_features), trainable=True)

        self.residual_1 = ResidualLayer(n_features, activation=activation)
        self.dense_before_skip = tf.keras.layers.Dense(n_features, activation=activation, use_bias=True)

        self.residual_3 = ResidualLayer(n_features, activation=activation)
        self.residual_4 = ResidualLayer(n_features, activation=activation)

    def call(self, inputs):
        m_l_1, e_rbf, a_sbf, id_kj, id_ji = inputs
        num_interactions = tf.shape(m_l_1)[0]
        
        m_ji = self.dense_ji(m_l_1)
        m_kj = self.dense_kj(m_l_1)

        e_rbf = self.dense_rbf(e_rbf)
        a_sbf = self.dense_sbf(a_sbf)

        m_kj = m_kj * e_rbf
        m_kj = tf.gather(m_kj, id_kj)
        m_kj = tf.einsum("wj,wl,ijl->wi", a_sbf, m_kj, self.bilin)
        x_kj = tf.math.unsorted_segment_sum(x_kj, id_ji, num_interactions)

        m_kj = m_ji + m_kj

        m_kj = self.residual_1(m_kj)
        m_kj = self.dense_before_skip(m_kj)
        
        m_kj = m_kj + m_l_1

        m_kj = self.residual_3(m_kj)
        m_kj = self.residual_4(m_kj)

        return m_kj