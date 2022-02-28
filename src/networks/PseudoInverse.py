
import numpy as np
import tensorflow as tf


class PseudoInverse(tf.keras.Model):
    def __init__(
        self, 
        image_shape, 
        uv, 
        op,
        measurement_weights=1,
        batch_size=20
        ):

        m_op = op()
        m_op.plan(uv, image_shape, (image_shape[0]*2, image_shape[1]*2), (6,6), batch_size=batch_size) #TODO change these hardcoded values


        inputs = tf.keras.Input([m_op.n_measurements], dtype=tf.complex64)
        outputs = tf.math.real(m_op.adj_op(inputs * measurement_weights))

        super().__init__(inputs=[inputs], outputs=outputs)
        self.compile(optimizer='adam', loss= tf.keras.losses.MSE)

    def fit(self, *args, **kwargs):
        return 