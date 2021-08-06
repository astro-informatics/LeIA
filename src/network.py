
# import segementation_models as sm
import tensorflow as tf
import numpy as np
from src.operators.measurement import NUFFT_op_TF

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class Gradient(tf.keras.layers.Layer):
    def __init__(self, m_op):
        self.m_op = m_op
        # self.data = tf.convert_to_tensor(data)
    
    # @tf.function(input_signature=[tf.TensorSpec( ], dtype=tf.float32)])
    # @tf.function(input_signature=[tf.TensorSpec( [None, 128,128,1], dtype=tf.float32), tf.TensorSpec( [None, 1, 4432], dtype=tf.complex64)])
    def __call__(self, x, y):
        # x = tf.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])
        x = tf.cast(x, tf.complex64)
        m = self.m_op.dir_op(x) 
        res = m -  tf.reshape(y, [1, -1, y.shape[1]])
        return tf.cast(self.m_op.adj_op( res ), tf.float32)[0, :,:,:, None]

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


class Gradient_2(tf.keras.layers.Layer):
    def __init__(self, m_op, shape, depth):
        self.m_op = m_op
        self.input_spec = [
            tf.keras.layers.InputSpec(
                dtype=tf.float32,
                shape=shape
            ),
            tf.keras.layers.InputSpec(
                dtype=tf.complex64,
                shape=shape
            )
        ]
        self.depth = depth +1
        self.trainable=False
    

    def __call__(self, x, y):
        x = tf.cast(x, tf.complex64)
        m = self.m_op.dir_op(x) 
        size = y.shape[1]
        y = y[
            :,
            size//2 - size//2**self.depth:size//2 + size//2**self.depth,
            size//2 - size//2**self.depth:size//2 + size//2**self.depth
            ] # only inner most part of y
        res = m -  y
        a = self.m_op.adj_op( res )
        return tf.cast(a, tf.float32)

class Unet(tf.keras.Model):
    def __init__(self, input_shape, uv, depth=2, start_filters=16, conv_layers=1, kernel_size=3, conv_activation='relu', output_activation='linear'):
        self.is_adapted=False
        inputs = tf.keras.Input(input_shape)
       
#         inputs2 = tf.keras.Input([len(uv),1], dtype=tf.complex64) # individual measurements
        inputs2 = tf.keras.Input(input_shape, dtype=tf.complex64) # fourier plane

        #TODO preprocessing (also on batch)
        #TODO upsampled FFT gradient
        
        x = inputs

        skips = []

        # construct gradient operators
        gradient_ops = []
        Nd = input_shape[0]
        Kd = input_shape[0]*2

        i= 0
#         op = NUFFT_op_TF(uv.T, Nd=(Nd//(2**i),Nd//(2**i)), Kd=(Kd//(2**i),Kd//(2**i)), Jd=(6,6))
#         gradient_layer = Gradient(op)

        op = fft_op()

        
#         with tf.name_scope("Grad"):
#             grad = gradient_layer(x[:,:,:,0], inputs2[:,:,0])

#         x = tf.keras.layers.Concatenate()([x,grad])
    

        #x = self.preprocess(inputs)
#         x = inputs
#         skips = []

        # convolution downward
        for i in range(depth):
            
            shape = (None, Nd//2**i, Nd//2**i)
            
            # gradients
            with tf.name_scope("Grad_" +str(i)):
    
                gradient_layer = Gradient_2(op, shape, i)

                grad = gradient_layer(x[:,:,:,0], inputs2[:,:,:,0])
                x = tf.keras.layers.Concatenate()([x,grad[:,:,:, None]])
            
            
            for j in range(conv_layers):
                x = tf.keras.layers.Conv2D(
                    filters=start_filters*2**(i), 
                    kernel_size=kernel_size, 
                    activation=conv_activation, 
                    padding='same',
                    name="conv2d_down_depth_" + str(i) + "_" + str(j)
                )(x)
                x = tf.keras.layers.BatchNormalization(
                    name="BatchNorm_down_depth_" + str(i) + "_" + str(j)
                )(x)
            skips.append(x)
            x = tf.keras.layers.MaxPool2D(padding='same')(x)


        # smallest layer
        for i in range(conv_layers):
            x = tf.keras.layers.Conv2D(
                    filters=start_filters*2**depth, 
                    kernel_size=kernel_size, 
                    activation=conv_activation, 
                    padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)



        # convolutions upward
        for i in range(depth):
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Concatenate()([x,skips[-(i+1)]])

            for j in range(conv_layers):
                x = tf.keras.layers.Conv2D(
                    filters=start_filters*2**(depth-(i+1)), 
                    kernel_size=kernel_size, 
                    activation=conv_activation, 
                    padding='same',
                    name="conv2d_up_depth_" + str(i) + "_" + str(j)
                )(x)
                x = tf.keras.layers.BatchNormalization(
                    name="BatchNorm_up_depth_" + str(i) + "_" + str(j)
                )(x)
                        
        # output formatting
        outputs = tf.keras.layers.Conv2D(
                    filters=1, 
                    kernel_size=1, 
                    padding='same',
                    activation=output_activation,
                    name="conv2d_output"
                    )(x)

        super().__init__(inputs=[inputs, inputs2], outputs=outputs)
    
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

