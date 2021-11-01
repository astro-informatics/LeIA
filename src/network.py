
# import segementation_models as sm
import tensorflow as tf
import numpy as np
from src.operators.measurement import NUFFT_op, NUFFT_op_TF

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class Gradient(tf.keras.layers.Layer):
    """
    Gradient operator
    TODO create docstring
    """
    def __init__(self, m_op, shape_x, shape_y, depth, learned=False):
        self.m_op = m_op
        self.learned = learned
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
    

    def __call__(self, x, y):
        x = tf.cast(x, tf.complex64)
        m = self.m_op.dir_op(x) 
        res = m -  y
        if not self.learned:
            grad = self.m_op.adj_op( res )
        else:
            grad = self.m_op.learned_adj_op( res )

        grad = tf.cast(grad, tf.float32)
        return grad

class fft_op():
    def __init__(self,):
        pass

    def dir_op(self, x):
        # zero pad in real space for upsampled k-space
        y = self._fft(x)
        return y
    
    def adj_op(self, y):
        x = self._ifft(y) # use only given visabilities for reconstruction
        return x
    
    def _fft(self, x):
        return tf.signal.fftshift(
            tf.signal.fft2d(
                tf.signal.fftshift(
                    x, axes=(-2,-1)
                )
            ), axes=(-2,-1))

    def _ifft(self, Fx):
        return tf.math.real(
            tf.signal.fftshift(
                tf.signal.ifft2d(
                    tf.signal.fftshift(
                        Fx, axes=(-2,-1)
                    )
                ), axes=(-2,-1)
            )
        )

class TF_nufft(NUFFT_op):
    """ Tensorfow adaptation of the nufft operator""" 
    def forward(self, x):
        return tf.numpy_function(self.dir_op, [x], tf.complex64)

    def adjoint(self, x):
        return tf.numpy_function(self.adj_op, [x], tf.complex64)

def conv_block(x, conv_layers, filters, kernel_size, activation, name):
    for j in range(conv_layers):
        x = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            activation=activation, 
            padding='same',
            name=name + "_conv2d_" + str(j)
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=name + "_batchnorm_" + str(j)
        )(x)
    return x 

def grad_block(x, measurements, x_dirty, grad_op, shape, name=""):
    with tf.name_scope(name + "_grad"):
        x_ = x[:,:,:,0] # select the first filter of x for gradient calculation
        gradi = grad_op(x_, measurements)
        gradi.set_shape(shape)
        x = tf.keras.layers.Concatenate()([
            x, 
            tf.expand_dims(x_dirty, axis=3),
            tf.expand_dims(gradi, axis=3)
            ]) # add gradient and dirty image at this scale
    return x


class Unet(tf.keras.Model):
    def __init__(self, 
        input_shape, 
        uv, 
        depth=2, 
        start_filters=16, 
        conv_layers=1, 
        kernel_size=3, 
        conv_activation='relu', 
        output_activation='linear', 
        grad=False, 
        learned_adjoint=False, 
        learned_grad=False, 
        grad_on_upsample=False
        ):

        batch_size = 20
        self.is_adapted=False
        # inputs = tf.keras.Input(input_shape)
        inputs = tf.keras.Input([len(uv)], dtype=tf.complex64) # individual measurements

        # inputs2 = tf.keras.Input(input_shape, dtype=tf.complex64) # fourier plane

        #TODO preprocessing (also on batch)
        #TODO upsampled FFT gradient
        
        x = inputs

        skips = []


        Nd = (input_shape[0], input_shape[1])
        Kd = (Nd[0]*2, Nd[1]*2)
        Jd = (6,6)

        op = NUFFT_op_TF()
        op.plan(uv, Nd, Kd, Jd, batch_size)

        # calculate density weighting
        grid_cell = 2*np.pi /512 
        binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
        binned = [tuple(x) for x in binned]
        cells = set(binned)
        w_gridded = np.zeros(uv.shape[0])
        for cell in list(cells):
            mask = np.all(np.array(cell) ==  binned, axis=1)
            w_gridded[mask] = np.sum(mask)

        # w = 
        w = 1/w_gridded
        w /= w.max()

        # get initial reconstruction through (learned) adjoint NUFFT
        if not learned_adjoint:
            x = tf.math.real(op.adj_op(inputs*w))
        else:
            x = tf.math.real(op.learned_adj_op(inputs*w))
        x_init = x
        x = tf.expand_dims(x, axis=3) # add empty dimension for CNNs

        if grad:
            # construct gradient operators
            gradient_ops = []
            subsampled_inputs = []
            dirty_images = []
    
            for i in range(depth):
                # m_op = TF_nufft() # numpy function tf operator
                m_op = NUFFT_op_TF() # TF native operator
                nd, kd = (Nd[0]//2**i, Nd[1]//2**i), (Kd[0]//2**i, Kd[1]//2**i)
                sel = np.linalg.norm(uv, axis=1) < np.pi / 2**i
                m_op.plan(uv[sel], nd, kd, Jd, batch_size)
                gradient_ops.append( 
                    Gradient(
                        m_op, [None, nd[0], nd[1]], [None, np.sum(sel)], i, learned=learned_grad
                    ) 
                )
                subsampled_input = tf.boolean_mask(inputs, sel, axis=1)           
                subsampled_inputs.append(subsampled_input)
                dirty_image = tf.math.real(m_op.adj_op(subsampled_input))
                dirty_images.append(dirty_image)
        #x = self.preprocess(inputs)

        # convolution downward
        for i in range(depth):
            
            shape = (None, Nd[0]//2**i, Nd[1]//2**i)
            
            if grad:
                # gradients
                x = grad_block(
                    x, 
                    measurements=subsampled_inputs[i],
                    x_dirty=dirty_images[i],  
                    grad_op=gradient_ops[i],
                    shape=shape, 
                    name=f"Down_depth_{i}"
                )

            x = conv_block(
                x, 
                conv_layers = conv_layers, 
                filters = start_filters*2**(i), 
                kernel_size=kernel_size, 
                activation=conv_activation,
                name=f"Down_depth_{i}"
            )
            skips.append(x)
            x = tf.keras.layers.MaxPool2D(padding='same')(x)


        if depth != 0:
            # smallest layer
            x = conv_block(
                    x, 
                    conv_layers = conv_layers, 
                    filters = start_filters*2**(depth), 
                    kernel_size=kernel_size, 
                    activation=conv_activation,
                    name=f"Down_depth_{depth}"
                )

        # convolutions upward
        for i in range(depth):
            shape = (None, Nd[0]//2**(depth-i-1), Nd[1]//2**(depth-i-1))

            x = tf.keras.layers.UpSampling2D()(x)

            if grad and grad_on_upsample:
                # gradients
                x = grad_block(
                    x, 
                    measurements=subsampled_inputs[-(i+1)], 
                    x_dirty=dirty_images[-(i+1)],  
                    grad_op=gradient_ops[-(i+1)],
                    shape=shape, 
                    name=f"Up_depth_{depth-i-1}"
                )

            
            x = tf.keras.layers.Concatenate()([x,skips[-(i+1)]])

            x = conv_block(
                x, 
                conv_layers = conv_layers, 
                filters = start_filters*2**(depth-(i+1)), 
                kernel_size=kernel_size, 
                activation=conv_activation,
                name=f"Up_depth_{depth-i-1}"
            )

        if depth != 0:                
            # output layer only if we have unet layers
            outputs = tf.keras.layers.Conv2D(
                        filters=1, 
                        kernel_size=1, 
                        padding='same',
                        activation=output_activation,
                        name="conv2d_output"
                        )(x)
            outputs = tf.squeeze(outputs) + x_init # remove extra dimension and add initial reconstruction
        else:
            outputs = tf.sqeeze(x)
        super().__init__(inputs=[inputs], outputs=outputs)

        self.compile(optimizer='adam', loss= tf.keras.losses.MSE)

#     def fit(self, x, y, **kwargs):
#         x = self.preprocess(x)
#         super().fit(x,y, **kwargs)

#     def predict(self, x, **kwargs):
#         x = self.preprocess(x)
#         pred = super().predict(x, **kwargs)
#         return pred

#     def evaluate(self, x, y, **kwargs):
#         x = self.preprocess(x)
#         return super().evaluate(x,y, **kwargs)
    
    def preprocess(self, x):
        if not self.is_adapted:
            self.mean = np.mean(x)
            self.std = np.std(x)
            self.is_adapted = True
        return (x-self.mean)/self.std
    
    def un_preprocess(self,x):
        if not self.is_adapted:
            return x
        return x*self.std + self.mean

def small_unet(input_shape=(256,256,1), uv=[], **kwargs):
    return Unet(
        input_shape,
        uv,
        **kwargs
    )

def medium_unet(input_shape=(256,256,1), uv=[], **kwargs):
    return Unet(
        input_shape=input_shape,
        uv=uv,
        depth=4, 
        start_filters=32, 
        conv_layers=2, 
        kernel_size=3, 
        **kwargs
    )

# def large_unet():
#     #TODO add preprocessing
#     model = sm.Unet('mobilenetv2', activation='relu', encoder_freeze=True, encoder_weights='imagenet')
#     return model

# unet = actual_unet()
# print(unet.summary())

