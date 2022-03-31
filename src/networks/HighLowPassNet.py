import tensorflow as tf
import numpy as np
from src.operators.NNFFT2D_TF import NNFFT2D_TF
  
class DownSample(tf.keras.layers.Layer):
    """Seperates a vector of measurements into high and low frequency measurements

    Args:
        tf ([type]): [description]
    """
    def __init__(self, low_freq_sel, depth):
        self.low_freq_sel = low_freq_sel # TODO, move the calculation of this to this function
        
        self.input_spec = [
            tf.keras.layers.InputSpec(
                dtype=tf.complex64,
                shape=[low_freq_sel.shape[0], 2]
            )
        ]
        self.depth = depth +1 # some 
    
    def __call__(self, full_freq_info):
        low_freq_info = tf.boolean_mask(
            full_freq_info, self.low_freq_sel, axis=1
            )    
        return full_freq_info, low_freq_info
 

class UpSample(tf.keras.layers.Layer):
    """Combines a lower-scale image and a higher-scale measurement vector into a higher-scale image

    Args:
        tf ([type]): [description]
    """
    def __init__(self, low_scale_op, high_scale_op, low_freq_sel, shape_x, depth, batch_size, weights=1):
        self.low_scale_op = low_scale_op # Low-scale operator
        self.high_scale_op = high_scale_op # High-scale operator
        self.low_freq_sel = low_freq_sel # boolean mask for sel of low freq signal
        self.w = weights
        
        indices = list(np.where(self.low_freq_sel)[0])
        self.indices = np.array([[i, indices[j]] for i in range(batch_size) for j in range(len(indices))])
        
        
        self.input_spec = [
            tf.keras.layers.InputSpec(
                dtype=tf.float32,
                shape=shape_x # Input low-scale image
            ),
            tf.keras.layers.InputSpec(
                dtype=tf.complex64,
                shape=[low_freq_sel.shape[0], 2]# Input high-scale freq info
            )
        ]
        self.depth = depth +1
        self.trainable=False
    

    def __call__(self, low_freq_image, high_freq_measurements, measurement_weights=1):
        low_freq_image = tf.cast(low_freq_image, tf.complex64)
        measurements = self.low_scale_op.dir_op(low_freq_image) # low-scale measurements

        full_freq_info = high_freq_measurements
        full_freq_info = tf.tensor_scatter_nd_update(
            full_freq_info,
            self.indices,
            tf.reshape(measurements, [-1])
        ) # update the lower coefficients with low-scale passed information
        
        up_sampled_image = self.high_scale_op.adj_op( full_freq_info * measurement_weights )
        up_sampled_image = tf.cast(up_sampled_image, tf.float32)

        return up_sampled_image

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


class HighLowPassNet(tf.keras.Model):
    def __init__(self, 
        image_shape, 
        uv, 
        op=None,
        depth=2, 
        start_filters=16,
        conv_layers=1, 
        measurement_weights=None,
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
        self.op = op
        assert not tf.executing_eagerly(), "HighLowPassNet cannot be run in eager execution mode, make sure to disable eager execution using `tf.compat.v1.disable_eager_execution()`"

        self.is_adapted=False
        # inputs = tf.keras.Input(input_shape)
        inputs = tf.keras.Input([len(uv)], dtype=tf.complex64) # individual measurements
        
        if measurement_weights is None:
            measurement_weights = np.ones(len(uv))
        x = inputs

        Nd = (image_shape[0], image_shape[1])
        Kd = (Nd[0]*2, Nd[1]*2)
        Jd = (6,6)

        x_ = inputs 

        ops = []
        low_freq_sels = []
        freq_weights = [measurement_weights] 

        new_uv = uv
        for i in range(depth+1):
            m_op = self.op() # TF native operator
            nd, kd = (Nd[0]//2**i, Nd[1]//2**i), (Kd[0]//2**i, Kd[1]//2**i)
            sel =  np.all(new_uv <  np.pi / 2**i, axis=1) # square selection (with exclusion region outside)

            new_uv = new_uv[sel]
            m_op.plan(new_uv*2**i, nd, kd, Jd, batch_size) # correct uv so they fill full plane of sub-sample

            freq_weights.append(freq_weights[i][sel])


            low_freq_sels.append(sel)
            ops.append(m_op)

        freq_weights = freq_weights[1:]

        
        x_init = tf.math.real(ops[0].adj_op(x_ * measurement_weights)) # initial reconstructions
        
        # calculate down and upscale layers
        down_layers = []
        up_layers =  []
        grad_layers = []
        for i in range(depth-1):
            print(i, (Nd[0]//2**i, Nd[1]//2**i), sum(low_freq_sels[i+1]))
            ds = DownSample(
                low_freq_sel=low_freq_sels[i+1], 
                depth=i
                )
            us = UpSample( 
                ops[i+1], 
                ops[i], 
                low_freq_sel=low_freq_sels[i+1], 
                shape_x=(Nd[0]//2**(i+1), 
                Nd[1]//2**(i+1)), 
                depth=i, 
                batch_size=self.batch_size,
                )

            down_layers.append(ds)
            up_layers.append(us)   

        for i in range(depth): 
            grad = Gradient(
                ops[i], 
                shape_x=(Nd[0]//2**(i),Nd[0]//2**(i)), 
                shape_y=(low_freq_sels[i].shape[0],2), 
                depth=i, 
                )
            grad_layers.append(grad) 


        freq_info = [x_]

        for i in range(depth-1):
            print(i, "down", x_.shape)
            with tf.name_scope("down_" + str(i)):
                high_freq_info, low_freq_info = down_layers[i](x_)
            freq_info.append(low_freq_info) # adding the high information to retain to a list

            x_ = low_freq_info

        # lowest layer:
        with tf.name_scope("adj_bottom"):
            x_ = ops[i+1].adj_op( x_  * freq_weights[i+1])
            x_ = tf.cast(x_, tf.float32)

        for i in range(depth-1, -1, -1):
            if i != depth-1:
                x_ = upsample(x_, up_layers, freq_info, i,               measurement_weights=freq_weights[i])
            print(i, "up", x_.shape)
            # x_ = tf.expand_dims(x_, axis=3)
            x_ = grad_block(x_, grad_layers, freq_info, i, freq_weights[i]) # calculate and concatenate gradients
            x_ = conv_block(x_, start_filters, i) # convolve and contract to 1 image

        # outputs = x_
        outputs = x_ + x_init # residual connection


        super().__init__(inputs=[inputs], outputs=outputs)

        self.compile(optimizer='adam', loss= tf.keras.losses.MSE)

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
        model = HighLowPassNet(
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

def grad_block(x_, grad_layers, freq_info, i, freq_weights=1):
    with tf.name_scope("grad_" + str(i)):
        
#         x_ = tf.stack([x_, grad ], axis=3)
        # dirty_im = tf.math.real(grad_layers[i].m_op.adj_op( freq_info[i] ))
        # grad = grad_layers[i](x_, freq_info[i])
        filtered_grad = grad_layers[i](x_, freq_info[i], measurement_weights=freq_weights)
        # x_ = tf.stack([x_, dirty_im, grad, filtered_grad], axis=3)
        x_ = tf.stack([x_, filtered_grad], axis=3)

        # x_ = tf.stack([x_, dirty_im, filtered_grad], axis=3)

        # x_ = tf.keras.layers.BatchNormalization()(x_)
    return x_

def upsample(x_, up_layers, freq_info, i, measurement_weights=1):
    with tf.name_scope("up_" + str(i)):
        x_ = up_layers[i]( x_, freq_info[i], measurement_weights=measurement_weights )
    return x_

def conv_block(x_, start_filters, i):
    with tf.name_scope("conv_" + str(i)):
        x_ = tf.keras.layers.Conv2D(
            filters=start_filters*2**(i),
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        )(x_)
        x_ = tf.keras.layers.BatchNormalization()(x_)
        x_ = tf.keras.layers.Conv2D(
            filters=start_filters*2**(i),
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        )(x_)
        x_ = tf.keras.layers.BatchNormalization()(x_)
        x_ = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            padding='same',
            activation='linear'
        )(x_)
        x_ = tf.squeeze(x_)
    return x_