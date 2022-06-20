import numpy as np
import tensorflow as tf


class UNet_var(tf.keras.Model):
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



        if input_type == "image":
            inputs = tf.keras.Input(image_shape, dtype=tf.float32)
            x = inputs
        elif input_type == "measurements":
            self.op = op
            self.m_op = self.op()
            self.m_op.plan(uv, image_shape, (image_shape[0]*2, image_shape[1]*2), (6,6), batch_size=batch_size) #TODO change these hardcoded values
            # assert m_op is not None, "Operator needs to be specified when passing measurements as input" 
            inputs = tf.keras.Input([self.m_op.n_measurements], dtype=tf.complex64)
            
            x = tf.math.real(self.m_op.adj_op(inputs * measurement_weights))

            
        else:
            raise ValueError("argument input_type should be one of ['image', 'measurements']")
        

        skips = []
  
        x_init = x

        tmp_max = tf.math.reduce_max(x, axis=(1,2))[:,None, None]
        tmp_min = tf.math.reduce_min(x, axis=(1,2))[:,None, None]

#         x = (x - tmp_min)/(tmp_max - tmp_min)

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

    
    def rebuild_with_op(self, uv):
        weigths = [self.layers[i].get_weights() for i in range(len(self.layers))]
        tf.keras.backend.clear_session()
        denoiser = UNet_var(
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
        
        for i in range(len(self.layers)):
            denoiser.layers[i].set_weights(weigths[i])
        
        # self.__dict__.update(denoiser.__dict__)
        # del denoiser
        return denoiser
            

    def fit_with_new_operator(self, x, y=None, uv=None, epochs=1, *args, **kwargs):
        self.rebuild_with_op(uv)
        self.fit(x,y, *args, epochs=epochs, **kwargs)
        
    @staticmethod
    def _convolutional_block(x, conv_layers, filters, **conv_kwargs):
        for j in range(conv_layers):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                **conv_kwargs
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
        return x