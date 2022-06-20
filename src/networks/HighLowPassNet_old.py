
# import segementation_models as sm
import tensorflow as tf
import numpy as np
from src.operators.measurement import NUFFT_op, NUFFT_op_TF

from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

class DownSample(tf.keras.layers.Layer):
    """
    Gradient operator
    TODO create docstring
    """
    def __init__(self, high_scale_op, low_scale_op, shape_x, low_freq_sel, depth):
        self.high_scale_op = high_scale_op
        self.low_scale_op = low_scale_op
        self.low_freq_sel = low_freq_sel
        
        self.input_spec = [
            tf.keras.layers.InputSpec(
                dtype=tf.float32,
                shape=shape_x
            )
        ]
        self.depth = depth +1
        self.trainable=False
    

    def __call__(self, x):
        x = tf.cast(x, tf.complex64)
        m = self.high_scale_op.dir_op(x) 

        high_freq_info = m
        low_freq_info = tf.boolean_mask(m, self.low_freq_sel, axis=1)    

        down_sampled_image = self.low_scale_op.adj_op( low_freq_info )
        down_sampled_image = tf.cast(down_sampled_image, tf.float32)
        return high_freq_info, down_sampled_image

class UpSample(tf.keras.layers.Layer):
    """
    Gradient operator
    TODO create docstring
    """
    def __init__(self, low_scale_op, high_scale_op, shape_x, low_freq_sel, depth, batch_size):
        self.low_scale_op = low_scale_op # Low-scale operator
        self.high_scale_op = high_scale_op # High-scale operator
        self.low_freq_sel = low_freq_sel # boolean mask for sel of low freq signal

        indices = list(np.where(self.low_freq_sel)[0])
        self.indices = np.array([[i, indices[j]] for i in range(batch_size) for j in range(len(indices))])
        
        
        self.input_spec = [
            tf.keras.layers.InputSpec(
                dtype=tf.float32,
                shape=shape_x # Input low-scale image
            ),
            tf.keras.layers.InputSpec(
                dtype=tf.complex64,
                shape=low_freq_sel.shape # Input high-scale freq info
            )
        ]
        self.depth = depth +1
        self.trainable=False
    

    def __call__(self, high_freq_measurements, low_freq_image):
        low_freq_image = tf.cast(low_freq_image, tf.complex64)
        measurements = self.low_scale_op.dir_op(low_freq_image) # low-scale measurements

        full_freq_info = high_freq_measurements
        full_freq_info = tf.tensor_scatter_nd_update(
            full_freq_info,
            self.indices,
            tf.reshape(measurements, [-1])
        )
        
        up_sampled_image = self.high_scale_op.adj_op( full_freq_info )

        up_sampled_image = tf.cast(up_sampled_image, tf.float32)
        return up_sampled_image

class UpSampleGrad(UpSample):
    """
    Gradient operator
    TODO create docstring
    """

    def __call__(self, high_freq_measurements, low_freq_image, true_measurements):
        low_freq_image = tf.cast(low_freq_image, tf.complex64)
        measurements = self.low_scale_op.dir_op(low_freq_image) # low-scale measurements

        full_freq_info = high_freq_measurements
        full_freq_info = tf.tensor_scatter_nd_update(
            full_freq_info,
            self.indices,
            tf.reshape(measurements, [-1])
        )
        
        up_sampled_image = self.high_scale_op.adj_op( full_freq_info )
        up_sampled_image = tf.cast(up_sampled_image, tf.float32)
        
        gradient = self.high_scale_op.adj_op( full_freq_info - true_measurements)
        gradient = tf.cast(gradient, tf.float32)
        return up_sampled_image, gradient


class HighLowPassNet(tf.keras.Model):
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
        with tf.name_scope("adjoint"):
            if not learned_adjoint:
                x = tf.math.real(op.adj_op(inputs*w))
            else:
                x = tf.math.real(op.learned_adj_op(inputs*w))
            x_init = x

            x_ = x



        ops = []
        low_freq_sels = []

        new_uv = uv
        for i in range(depth+1):
            m_op = NUFFT_op_TF() # TF native operator
            nd, kd = (Nd[0]//2**i, Nd[1]//2**i), (Kd[0]//2**i, Kd[1]//2**i)
            sel =  np.all(new_uv <  np.pi / 2**i, axis=1) # square selection (with exclusion region outside)

            new_uv = new_uv[sel]
            m_op.plan(new_uv*2**i, nd, kd, Jd, batch_size) # correct uv so they fill full plane of sub-sample

            low_freq_sels.append(sel)
            ops.append(m_op)
        
        
        
        # calculate down and upscale layers
        down_layers = []
        up_layers =  []
        for i in range(depth):
            print(i, (Nd[0]//2**i, Nd[1]//2**i), sum(low_freq_sels[i+1]))
            ds = DownSample( ops[i], ops[i+1], shape_x=(Nd[0]//2**i, Nd[1]//2**i), low_freq_sel=low_freq_sels[i+1], depth=i)
#             us = UpSample( ops[i+1], ops[i], shape_x=(Nd[0]//2**(i+1), Nd[1]//2**(i+1)), low_freq_sel=low_freq_sels[i+1], depth=i, batch_size=20)
            us = UpSampleGrad( ops[i+1], ops[i], shape_x=(Nd[0]//2**(i+1), Nd[1]//2**(i+1)), low_freq_sel=low_freq_sels[i+1], depth=i, batch_size=20)

            down_layers.append(ds)
            up_layers.append(us)    

        up_info = []

        subsampled_inputs = [ops[0].dir_op(x_)*w] # TODO check wethter this filtering helps

        for i in range(1, depth):
            subsampled_inputs.append(tf.boolean_mask(subsampled_inputs[i-1], low_freq_sels[i], axis=1))


        for i in range(len(down_layers)):
            print(i, "down", x_.shape)
            with tf.name_scope("down_" + str(i)):
                h, im = down_layers[i](x_)
            up_info.append(h) # adding the high information to retain to a list
            x_ = im

        for i in range(len(up_layers)):
            print(i, "up", x_.shape)
            h = up_info[-(i+1)]
        #     x_ = up_layers[-(i+1)](h,x_)
            with tf.name_scope("up_" + str(i)):
                x_, grad = up_layers[-(i+1)](h,x_, subsampled_inputs[-(i+1)])
#                 x_ = up_layers[-(i+1)](h, x_)
            with tf.name_scope("conv_" + str(i)):
                x_ = tf.expand_dims(x_, axis=3) # add empty dimension for CNNs
                grad = tf.expand_dims(grad, axis=3) # add empty dimension for CNNs

                x_ = tf.keras.layers.Concatenate()([x_, grad])
                x_ = tf.keras.layers.Conv2D(
                    filters=start_filters*2**(depth-i-1),
                    kernel_size=(3,3),
                    padding='same',
                    activation='relu'
                )(x_)
                x_ = tf.keras.layers.BatchNormalization()(x_)
                x_ = tf.keras.layers.Conv2D(
                    filters=start_filters*2**(depth-i-1),
                    kernel_size=(3,3),
                    padding='same',
                    activation='relu'
                )(x_)
                x_ = tf.keras.layers.BatchNormalization()(x_)
                x_ = tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(1,1),
                    padding='same',
                    activation='sigmoid'
                )(x_)
                x_ = tf.squeeze(x_)

        outputs = x_

        super().__init__(inputs=[inputs], outputs=outputs)

        self.compile(optimizer='adam', loss= tf.keras.losses.MSE)