
import numpy as np
import tensorflow as tf

from src.networks.UNet import UNet
# from src.networks.PseudoInverse import PsuedoInverse


class Gradient(tf.keras.layers.Layer):
    """
    Gradient operator
    TODO create docstring
    """
    def __init__(self, m_op, shape_x, shape_y, depth):
        self.m_op = m_op
        self.input_spec = [
            tf.keras.layers.InputSpec(
                dtype=tf.float32,
                shape=shape_x
            ),
            tf.keras.layers.InputSpec(
                dtype=tf.complex64,
                shape=shape_y
            )
        ]
        self.depth = depth +1
        self.trainable=False
    

    def __call__(self, x, y, measurement_weights=1):
        x = tf.cast(x, tf.complex64)
        m = self.m_op.dir_op(x) 
        res = m -  y
        grad = self.m_op.adj_op( res  * measurement_weights)
        grad = tf.cast(grad, tf.float32)
        return grad


class LFB(tf.keras.Model):
    def __init__(
        self, 
        image_shape, 
        uv, 
        op=None,
        measurement_weights=1,
        batch_size=20
        ):

        m_op = op()
        m_op.plan(uv, image_shape, (image_shape[0]*2, image_shape[1]*2), (6,6), batch_size=batch_size) #TODO change these hardcoded values

        # pseudoinverse = PsuedoInverse(image_shape, uv, op=op, measurement_weights=measurement_weights)

        inputs = tf.keras.Input([m_op.n_measurements], dtype=tf.complex64)
        x = tf.math.real(m_op.adj_op(inputs * measurement_weights))


        unet = UNet(image_shape, uv, op=op, depth=2, conv_layers=2, input_type='image', batch_size=batch_size)

        gradient = Gradient(m_op, image_shape, m_op.n_measurements, 0)

        lr = tf.keras.layers.Dense(1, use_bias=False)
        for i in range(7):
            x = x + lr( gradient(x, inputs))
            x = unet(x)
        outputs = x

        super().__init__(inputs=[inputs], outputs=outputs)

        self.compile(optimizer='adam', loss= tf.keras.losses.MSE)
