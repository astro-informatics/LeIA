
import numpy as np
import tensorflow as tf

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
        uv, 
        op=None,
        depth=2, 
        start_filters=16,
        measurement_weights=1,
        conv_layers=1, 
        kernel_size=3, 
        output_activation='linear', 
        input_type="measurements",
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
        self.op = op
        self.residual = residual

        assert not tf.executing_eagerly(), "GUNet cannot be run in eager execution mode, make sure to disable eager execution using `tf.compat.v1.disable_eager_execution()`"
        if self.measurement_weights is None:
            self.measurement_weights = np.ones(len(uv))

        Nd = (image_shape[0], image_shape[1])
        Kd = (Nd[0]*2, Nd[1]*2)
        Jd = (6,6)

        ops = [] # list with measurement operators for each scale
        low_freq_sels = [] # list with frequency measurements for each scale
        freq_weights = [self.measurement_weights]  # list with weights for each scale
        grad_layers = []
        subsampled_inputs = []

        for i in range(depth+1):
            m_op = self.op() # TF native operator
            nd, kd = (Nd[0]//2**i, Nd[1]//2**i), (Kd[0]//2**i, Kd[1]//2**i)
            sel =  np.all(uv <  np.pi / 2**i, axis=1) # square selection (with exclusion region outside)

            new_uv = uv[sel]
            m_op.plan(new_uv*2**i, nd, kd, Jd, batch_size) # correct uv so they fill full plane of sub-sample

            freq_weights.append(freq_weights[0][sel])


            low_freq_sels.append(sel)
            ops.append(m_op)

        freq_weights = freq_weights[1:]



        if input_type == "image":
            inputs = tf.keras.Input(image_shape, dtype=tf.float32)
            x = inputs
        elif input_type == "measurements":
            assert op is not None, "Operator needs to be specified when passing measurements as input" 
            inputs = tf.keras.Input([ops[0].n_measurements], dtype=tf.complex64)
            x = tf.math.real(ops[0].adj_op(inputs * freq_weights[0]))
        else:
            raise ValueError("argument input_type should be one of ['image', 'measurements']")

        for i in range(depth+1): 
            subsampled_input = tf.boolean_mask(inputs, low_freq_sels[i], axis=1)
            subsampled_input.set_shape([batch_size, np.sum(low_freq_sels[i])])         
            subsampled_inputs.append(subsampled_input)

            grad = Gradient(
                ops[i], 
                shape_x=(Nd[0]//2**(i),Nd[0]//2**(i)), 
                shape_y=(subsampled_inputs[i].shape[0],2), 
                depth=i, 
                )
            grad_layers.append(grad)


        skips = []
  
        x_init = x
        x = tf.expand_dims(x, axis=3) # add empty dimension for CNNs

        conv_kwargs = {
            "kernel_size": kernel_size,
            "activation": "relu",
            "padding": "same",
            }

        # x = self._grad_block(x, grad_layers, subsampled_inputs[0], 0, freq_weights[0]) # calculate and concatenate gradients
        x = tf.keras.layers.Conv2D(filters=start_filters, **conv_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # convolution downward
        for i in range(depth):
            if i != 0:
                x = self._grad_block(x, grad_layers, subsampled_inputs[i], i, freq_weights[i])     
     
            x = self._convolutional_block(
                x, 
                conv_layers = conv_layers, 
                filters = start_filters*2**(i), # #filters increase with depth
                **conv_kwargs
            )
            skips.append(x)
            x = tf.keras.layers.MaxPool2D(padding='same')(x)


        # Lowest scale
        x = self._grad_block(x, grad_layers, subsampled_inputs[depth], depth, freq_weights[depth])
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
            if i != 0:
                x = self._grad_block(x, grad_layers, subsampled_inputs[depth-(i+1)], depth-(i+1), freq_weights[depth-(i+1)])            

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

    @staticmethod
    def _grad_block(x_, grad_layers, y, i, freq_weights=1):
        with tf.name_scope("grad_" + str(i)):
            dirty_im = tf.expand_dims(tf.math.real(grad_layers[i].m_op.adj_op(  y)), axis=3)

            filtered_grad = tf.expand_dims(grad_layers[i](x_[:,:,:,0], y, measurement_weights=freq_weights), axis=3)
            # if the weights are not uniform, add a non-weighted gradient
            if np.any(freq_weights != np.ones(len(freq_weights))):
                grad = tf.expand_dims(grad_layers[i](x_[:,:,:,0], y, measurement_weights=1), axis=3)
                # return tf.concat([x_[:,:,:,:], grad, filtered_grad], axis=3)
                # return tf.concat([x_[:,:,:,:], dirty_im, grad, filtered_grad], axis=3)
                x_ = tf.concat([x_[:,:,:,:], grad, filtered_grad, dirty_im], axis=3)
                x_ = tf.keras.layers.Conv2D(16*2**(4-(i+1)), kernel_size=3, padding='same', activation='relu')(x_) #TODO remove hardcoded depth=4
                return x_

            else: 
                return tf.concat([x_[:,:,:,:], filtered_grad], axis=3)

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