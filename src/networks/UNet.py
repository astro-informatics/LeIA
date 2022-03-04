
import numpy as np
import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(
        self, 
        image_shape, 
        uv, 
        op=None,
        depth=2, 
        start_filters=16,
        measurement_weights=1,
        conv_layers=1, 
        kernel_size=3, 
        output_activation='linear', 
        input_type="image",
        batch_size=20
        ):



        if input_type == "image":
            inputs = tf.keras.Input(image_shape, dtype=tf.float32)
            x = inputs
        elif input_type == "measurements":
            m_op = op()
            m_op.plan(uv, image_shape, (image_shape[0]*2, image_shape[1]*2), (6,6), batch_size=batch_size) #TODO change these hardcoded values
            assert op is not None, "Operator needs to be specified when passing measurements as input" 
            inputs = tf.keras.Input([m_op.n_measurements], dtype=tf.complex64)
            x = tf.math.real(m_op.adj_op(inputs * measurement_weights))
        else:
            raise ValueError("argument input_type should be one of ['image', 'measurements']")
        

        skips = []
  
        x_init = x
        x = tf.expand_dims(x, axis=3) # add empty dimension for CNNs

        conv_kwargs = {
            "kernel_size": kernel_size,
            "activation": "relu",
            "padding": "same",
            }

        x = tf.keras.layers.Conv2D(filters=start_filters, **conv_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # convolution downward
        for i in range(depth):            
            x = self._convolutional_block(
                x, 
                conv_layers = conv_layers, 
                filters = start_filters*2**(i), # #filters increase with depth
                **conv_kwargs
            )
            skips.append(x)
            x = tf.keras.layers.MaxPool2D(padding='same')(x)


        # Lowest scale
        x = self._convolutional_block(
                x, 
                conv_layers = conv_layers, 
                filters = start_filters*2**(depth), 
                **conv_kwargs
            )

        # convolutions upward
        for i in range(depth):
            x = tf.keras.layers.Conv2DTranspose(
                filters=start_filters*2**(depth-(i+1)), # #filters increase with depth
                strides=(2,2),
                **conv_kwargs
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # x = tf.keras.layers.ReLU()(x)
            
            x = tf.keras.layers.Concatenate()([x,skips[-(i+1)]])

            x = self._convolutional_block(
                x, 
                conv_layers = conv_layers, 
                filters = start_filters*2**(depth-(i+1)), 
                **conv_kwargs
            )

        x = tf.keras.layers.Conv2D(
                    filters=1, 
                    kernel_size=1, 
                    padding='same',
                    activation=output_activation,
                    name="conv2d_output"
                    )(x)
        outputs = tf.squeeze(x, axis=-1) + x_init # remove extra dimension and add initial reconstruction

        super().__init__(inputs=[inputs], outputs=outputs)
        print(inputs, outputs)
        self.compile(optimizer='adam', loss= tf.keras.losses.MSE)

    @staticmethod
    def _convolutional_block(x, conv_layers, filters, **conv_kwargs):
        for j in range(conv_layers):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                **conv_kwargs
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
        return x