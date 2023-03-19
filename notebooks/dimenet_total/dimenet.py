import tensorflow as tf
import scipy.sparse as sp
import numpy as np

from ..layers.embedding_layer import EmbeddingLayer
from ..layers.interaction_layer import InteractionLayer
from ..layers.output_layer import OutputLayer
from ..layers.rbf_layer import RbfLayer
from ..layers.residual_layer import ResidualLayer
from ..layers.sbf_layer import SbfLayer

from .activation import swish

class DimeNet(tf.keras.Model):
    def __init__(self, n_features=128, N_bil=8, N_shbf=7, N_srbf=6, N_rbf=6,
                 c=5.0, activation=swish, name="dimenet", **kwargs):
        super().__init__(name=name, **kwargs)

        self.cutoff = c
        
        self.rbf_layer = RbfLayer(N_rbf, c)
        self.sbf_layer = SbfLayer(N_shbf, N_shbf, c)
        self.embedding_layer = EmbeddingLayer(n_features, activation=activation)

        self.output_layers = [OutputLayer(n_features, activation) for _ in range(8)]
        self.interaction_layers = [InteractionLayer(n_features, N_bil, activation=activation) for _ in range(7)]

    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        # ReLU prevents negative numbers in sqrt
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1)))
        return Dij

    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets"""
        Ri = tf.gather(R, id3_i)
        Rj = tf.gather(R, id3_j)
        Rk = tf.gather(R, id3_k)
        R1 = Rj - Ri
        R2 = Rk - Ri  # This should actually be `Rk - Rj`. Change it if you're not using a pretrained model, since the correct version performs better.
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle
    
    def call(self, inputs):
        Z, R                         = inputs['Z'], inputs['R']
        batch_seg                    = inputs['batch_seg']
        idnb_i, idnb_j               = inputs['idnb_i'], inputs['idnb_j']
        id_expand_kj, id_reduce_ji   = inputs['id_expand_kj'], inputs['id_reduce_ji']
        id3dnb_i, id3dnb_j, id3dnb_k = inputs['id3dnb_i'], inputs['id3dnb_j'], inputs['id3dnb_k']
        #n_atoms = tf.shape(Z)[0]

        # Calculate distances
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        Anglesijk = self.calculate_neighbor_angles(
            R, id3dnb_i, id3dnb_j, id3dnb_k)
        
        sbf = self.sbf_layer([Dij, Anglesijk])

        m = self.embedding_layer([Z, rbf, idnb_i, idnb_j])
        P = self.output_layers[0]([m, rbf, idnb_i])
        
        for i in range(7):
            m = self.interaction_layers[i]([m, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_layers[i+1]([m, rbf, idnb_i])

        P = tf.math.segment_sum(P, batch_seg)
        return 