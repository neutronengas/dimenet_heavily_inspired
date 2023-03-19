import tensorflow as tf
import numpy as np

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, n_features, activation=None, name="embedding"):
        super().__init__(name=name)
        self.n_features = n_features
    
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))
        # Aspirin only contains C, H, O atoms, hence 3 embedding vectors
        self.embeddings = self.add_weight(name="embeddings", shape=(3, self.n_features), 
                                          dtype=np.float32, initializer=emb_init, trainable=True)

        self.dense_e_rbf = tf.keras.layers.Dense(self.n_features, activation=activation, 
                                                 use_bias=False, kernel_initializer="glorot_uniform")

        self.dense_final = tf.keras.layers.Dense(self.n_features, activation=activation,
                                                 use_bias=True, kernel_initializer="glorot_uniform")
        
    def call(self, input):
        z, e_rbf, id_i, id_j = input
        z_i = tf.gather(z, id_i)
        z_j = tf.gather(z, id_j)
        h_i = tf.gather(self.embeddings, z_i)
        h_j = tf.gather(self.embeddings, z_j)
        e_rbf = self.dense_e_rbf(e_rbf)

        return self.dense_final(tf.concat([h_i, h_j, e_rbf]), axis=-1)
            