import tensorflow as tf

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, n_features, activation=None, name="output"):
        super().__init__(name=name)

        self.dense_rbf = tf.keras.layers.Dense(n_features, use_bias=False)

        self.dense_1 = tf.keras.layers.Dense(n_features, use_bias=False)
        self.dense_2 = tf.keras.layers.Dense(n_features, use_bias=False)
        self.dense_3 = tf.keras.layers.Dense(n_features, use_bias=False)
        self.final = tf.keras.layers.Dense(2, use_bias=False)

    def call(self, inputs):
        m, e_rbf, id_i = inputs
        
        e_rbf = self.dense_rbf(e_rbf)
        m = e_rbf * m
        m_j = tf.gather_nd(m, id_i)
        m = tf.reduce_sum(m_j, axis=0)

        for l in [self.dense_1, self.dense_2, self.dense_3]:
            m = l(m)
        
        m = self.final(m)
        return m