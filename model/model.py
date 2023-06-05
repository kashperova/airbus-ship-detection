"""
Script for U-net implementation
"""

import tensorflow as tf


class DoubleConv(tf.keras.Model):
    """
    Double convolutional block of U-net architecture
    Contains two convolutional layers followed by batch normalization
    and LeakyReLU activation functions.

    __Attributes__
    in_ch: int - input channels
    out_ch: int - output channels
    height: int - picture height
    width: int - picture width
    conv: tf.keras.Sequential - sequence of convolutional layers
    """

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                out_ch, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(
                out_ch, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()])

    def call(self, x):
        x = self.conv(x)
        return x


class InConv(tf.keras.Model):
    """
    Input block - wrapper for DoubleConv class

    __Attributes__
    in_ch: int - input channels
    out_ch: int - output channels
    conv: DoubleConv - class of double convolutional block
    """

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def call(self, x):
        x = self.conv(x)
        return x


class Down(tf.keras.Model):
    """
    Encoder block.
    Decreases dimension by two and increases the number of channels
    using together MaxPolling layer and layer of previous DoubleConv class

     __Attributes__
    in_ch: int - input channels
    out_ch: int - output channels
    encode: tf.keras.Sequential - encoder block
    """

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.encode = tf.keras.Sequential([tf.keras.layers.MaxPooling2D((2, 2)),
                                          DoubleConv(in_ch, out_ch)])

    def call(self, x):
        x = self.encode(x)
        return x


class Up(tf.keras.Model):
    """
    Decoder block.
    Each up block takes the output of the previous layer and upscales it
    using trasnpose convolution layer (dimension is increased by a factor of 2)

    Output of the corresponding x2 layer is concatinated with x1 layer
    and the result is passed through DoubleConv layer.

     __Attributes__
    in_ch: int - input channels
    out_ch: int - output channels
    up: tf.keras.Sequential - upscaling layer
    conv: tf.keras.Sequential - double convolutional layer
    """

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = tf.keras.layers.Conv2DTranspose(filters=in_ch // 2, kernel_size=(2, 2), strides=(2, 2),
                                                )
        self.conv = DoubleConv(in_ch, out_ch)

    def call(self, x1, x2):
        x1 = self.up(x1)
        x = tf.keras.layers.concatenate([x2, x1])
        x = self.conv(x)
        return x


class OutConv(tf.keras.Model):
    """
    Output layer - convolutional layer with kernel_size = (1,1) and sigmoid activation.
    Decrease number of channels and pass the result though sigmoid activation function.

     __Attributes__
    in_ch: int - input channels
    out_ch: int - output channels
    conv: tf.keras.Conv2D - convolutional layer
    """

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1, 1),
                                           activation='sigmoid')

    def call(self, x):
        x = self.conv(x)
        return x


class UNet(tf.keras.Model):
    """
    Main class for U-net implementation.

    in_ch for encoder blocks: 3 -> 64 -> 128 -> 256 -> 512 -> 512
    out_ch for decoder blocks: 256 -> 128 -> 64 -> 64 -> 1

    height, width for encoder blocks: 256 -> 128 -> 64 -> 32 -> 16
    height, width for decoder blocks: 16 -> 32 -> 64 -> 128 -> 256

     __Attributes__
    in_channels: int - input channels
    num_classes: int - amount of target classes
                        (2 - ship or not ship in our task)
    inc: InConv class - first layer for increasing channels
    down1, down2, down3, down4: Down class - encoder blocks
    up1, up2, up3, up4: Up class - decoder blocks
    outc: OutConv class - last layer for predicting
    """

    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_classes)

    def call(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x