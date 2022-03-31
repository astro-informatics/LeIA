
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
        residual = True,
        batch_size=20
        ):

        # store parameters
        self.image_shape = image_shape
        self.uv = uv
        self.depth = depth
        self.start_filters = start_filters
        self.measurement_weights = measurement_weights
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size 
        self.output_activation = output_activation
        self.input_type = input_type
        self.batch_size = batch_size
        self.residual = residual

        # check input type
        if input_type == "image":
            inputs = tf.keras.Input(image_shape, dtype=tf.float32)
            x = inputs
        elif input_type == "measurements":
            self.op = op # save the used operator
            m_op = self.op()
            m_op.plan(uv, image_shape, (image_shape[0]*2, image_shape[1]*2), (6,6), batch_size=batch_size) #TODO change these hardcoded values for upsampling
            assert op is not None, "Operator needs to be specified when passing measurements as input" 
            # calculate initial image using a weighted adjoint
            inputs = tf.keras.Input([m_op.n_measurements], dtype=tf.complex64)
            x = tf.math.real(m_op.adj_op(inputs * measurement_weights))
        else:
            raise ValueError("argument input_type should be one of ['image', 'measurements']")
        
        skips = []

        conv_kwargs = {
            "kernel_size": kernel_size,
            "activation": "relu",
            "padding": "same",
            }
        x_init = x
        x = tf.expand_dims(x, axis=3)
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

        # remove extra dimension and add initial reconstruction
        if residual:
            outputs = tf.squeeze(x, axis=-1) + x_init 
        else:
            outputs = tf.squeeze(x, axis=-1)

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

    def rebuild_with_op(self, uv):
        """Rebuilds the current network with a new sampling distribution

        Args:
            uv : new sampling distribution

        Returns:
            model : model rebuild with new sampling distribution
        """
        # extract weights from current model
        weigths = [self.layers[i].get_weights() for i in range(len(self.layers))]

        # reset graph and make new model with same parameters but new sampling distribution
        tf.keras.backend.clear_session()
        model = UNet(
            self.image_shape, 
            uv,
            op=self.op, 
            depth=self.depth, 
            start_filters=self.start_filters,
            conv_layers=self.conv_layers, 
            kernel_size=self.kernel_size, 
            output_activation=self.output_activation, 
            input_type=self.input_type, 
            measurement_weights=self.measurement_weights,
            batch_size=self.batch_size
        )
        
        # transfer old weights to new model
        for i in range(len(self.layers)):
            model.layers[i].set_weights(weigths[i])
        
        return model