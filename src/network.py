

import tensorflow as tf



class Unet(tf.keras.Model):
    def __init__(self, input_shape, depth=2, start_filters=16, conv_layers=1, kernel_size=3, conv_activation='relu'):

        inputs = tf.keras.Input(input_shape)
        x = inputs

        skips = []

        # convolution downward
        for i in range(depth):
            for j in range(conv_layers):
                x = tf.keras.layers.Conv2D(
                    filters=start_filters*2**(i), 
                    kernel_size=kernel_size, 
                    activation=conv_activation, 
                    padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
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
                    padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                        
        # output formatting
        outputs = tf.keras.layers.Conv2D(
                    filters=1, 
                    kernel_size=1, 
                    padding='same')(x)

        super().__init__(inputs=inputs, outputs=outputs)
    
        self.compile(optimizer='adam', loss= tf.keras.losses.MSE)


def small_unet(input_shape=(128,128,1)):
    return Unet(input_shape)

def medium_unet(input_shape=(256,256,1)):
    return Unet(
        input_shape=input_shape,
        depth=3, 
        start_filters=32, 
        conv_layers=2, 
        kernel_size=3, 
        conv_activation='relu'
    )

# unet = actual_unet()
# print(unet.summary())
