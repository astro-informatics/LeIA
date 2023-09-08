import numpy as np
import tensorflow as tf

class Gradient(tf.keras.layers.Layer):
    """
    Gradient operator
    TODO create docstring
    """
    def __init__(self, psf, shape_x, shape_y, depth):
        self.psf = tf.cast(psf, tf.complex64)
        
        self.ft_psf = self._fft(self.psf)

        # TODO this crop should probably be replaced with a partition of unity filter
        self.ft_psf = self.ft_psf[
            :, 
            psf.shape[1]//2 - shape_x[0]//2:psf.shape[1]//2 - shape_x[0]//2 + shape_y[0], 
            psf.shape[2]//2 - shape_x[1]//2:psf.shape[2]//2 - shape_x[1]//2 + shape_y[0]
            ] # crop to appropriate size in fourier space
        # print(depth, shape_x, shape_y, self.ft_psf.shape)
        
        self.input_spec = [
            tf.keras.layers.InputSpec(
                dtype=tf.float32,
                shape=shape_x
            ),
            tf.keras.layers.InputSpec(
                dtype=tf.float32,
                shape=shape_y
            )
        ]
        self.depth = depth +1
        self.trainable=False
    
    @staticmethod
    def _ifft(kk):
        """from 2d fourier space to 2d image space"""
        return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(kk, axes=(-2,-1))), axes=(-2,-1)) 

    @staticmethod
    def _fft(xx):
        """from 2d fourier space to 2d image space"""
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(xx, axes=(-2,-1))), axes=(-2,-1))
    
    def forward(self, x):
        """convolution with psf calculated in fourier space"""
        return self._ifft(self._fft(x) * self.ft_psf)

    def adjoint(self, y):
        """auto-correlation with psf calculated in fourier space"""
        return self._ifft(self._fft(y) * tf.math.conj(self.ft_psf)) #TODO conj of psf or conj of y?

    def __call__(self, x, y):
        x = tf.cast(x, tf.complex64)
        y = tf.cast(y, tf.complex64)
        m = self.forward(x)
        res = m -  y
        grad = self.adjoint( res )
        grad = tf.cast(grad, tf.float32)
        return grad

def conv_block(x, conv_layers, filters, kernel_size, activation, name):
    for j in range(conv_layers):
        x = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            # activation=activation, 
            padding='same',
            name=name + "_conv2d_" + str(j)
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=name + "_batchnorm_" + str(j)
        )(x)
        x = tf.keras.layers.ReLU()(x)
    return x 


class GUnet(tf.keras.Model):
    def __init__(
        self, 
        image_shape, 
        depth=2, 
        start_filters=16,
        conv_layers=1, 
        kernel_size=3, 
        output_activation='linear', 
        residual = True,
        batch_size=20
        ):

        # store parameters
        self.image_shape = image_shape
        self.depth = depth
        self.start_filters = start_filters
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size 
        self.output_activation = output_activation
        self.batch_size = batch_size
        self.residual = residual

        Nd = (image_shape[0], image_shape[1])

        grad_layers = []
        subsampled_inputs = []

        dirty_image = tf.keras.Input(image_shape, dtype=tf.float32) # dirty image input
        psf = tf.keras.Input(image_shape, dtype=tf.float32) # psf input

        for i in range(depth+1): 
            subsampled_input = dirty_image[:, ::2**i, ::2**i] # subsampled dirty image for subsampled gradients    
            subsampled_inputs.append(subsampled_input)

            grad = Gradient(
                psf, 
                shape_x=(Nd[0]//2**(i),Nd[0]//2**(i)), 
                shape_y=(subsampled_inputs[i].shape[1:]), 
                depth=i, 
                )
            grad_layers.append(grad)

        skips = []
  
        x = dirty_image
        x_init = x
        x = tf.expand_dims(x, axis=3) # add empty dimension for CNNs

        conv_kwargs = {
            "kernel_size": kernel_size,
            "activation": "relu",
            "padding": "same",
            }

        x = self._grad_block(x, grad_layers, subsampled_inputs[0], 0, conv_filters=start_filters) # calculate and concatenate gradients
        x = tf.keras.layers.Conv2D(filters=start_filters, **conv_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # convolution downward
        for i in range(depth):
            if i != 0:
                x = self._grad_block(x, grad_layers, subsampled_inputs[i], i, conv_filters=start_filters*2**(i-1))     
     
            x = self._convolutional_block(
                x, 
                conv_layers = conv_layers, 
                filters = start_filters*2**(i), # #filters increase with depth
                **conv_kwargs
            )
            skips.append(x)
            x = tf.keras.layers.MaxPool2D(padding='same')(x)


        # Lowest scale
        x = self._grad_block(x, grad_layers, subsampled_inputs[depth], depth, conv_filters=start_filters*2**(depth-1))
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

            # if i == depth-1:
            
            x = self._grad_block(x, grad_layers, subsampled_inputs[depth-(i+1)], depth-(i+1), conv_filters=start_filters*2**(depth-(i+1)))            

            x = tf.keras.layers.Concatenate()([x,skips[depth-(i+1)]])

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
        if self.residual:
            outputs = tf.squeeze(x, axis=-1) + x_init 
        else:
            outputs = tf.squeeze(x, axis=-1)


        super().__init__(inputs=[dirty_image, psf], outputs=outputs)
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

    @staticmethod
    def _grad_block(x_, grad_layers, dirty_im, i, conv_filters=16,  freq_weights=1):
        with tf.name_scope("grad_" + str(i)):
            grad = tf.expand_dims(grad_layers[i](x_[:,:,:,0], dirty_im), axis=3)
    
        x_ = tf.concat([x_[:,:,:,:], grad, tf.expand_dims(dirty_im, axis=3)], axis=3) # TODO see if we want to remove y
        x_ = tf.keras.layers.Conv2D(conv_filters, kernel_size=3, padding='same', activation='relu')(x_) # remove hardcoded start_filters 16

        return x_

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
        model = GUnet(
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
            batch_size=self.batch_size,
            residual=self.residual
        )
        
        # transfer old weights to new model
        for i in range(len(self.layers)):
            model.layers[i].set_weights(weigths[i])
        
        return model
