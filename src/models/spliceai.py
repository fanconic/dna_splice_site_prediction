import tensorflow as tf

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, N, W, D):
        super(ResBlock, self).__init__()
        self.BN_1 = tf.keras.layers.BatchNormalization()
        self.BN_2 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv1D(N, W, dilation_rate=D, padding = "same")
        self.conv_2 = tf.keras.layers.Conv1D(N, W, dilation_rate=D, padding = "same")
        
    def call(self, inputs, training=None):
        x = self.BN_1(inputs, training)
        x = tf.keras.activations.relu(x)
        x = self.conv_1(x)
        x = self.BN_2(x, training)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
            
        return x + inputs


class SpliceAI80(tf.keras.Model):

    def __init__(self):
        super(SpliceAI80, self).__init__()
        self.conv_1 = tf.keras.layers.Conv1D(32, 1, dilation_rate=1)
        self.conv_2 = tf.keras.layers.Conv1D(32, 1, dilation_rate=1)
        self.conv_3 = tf.keras.layers.Conv1D(32, 1, dilation_rate=1)
        self.conv_4 = tf.keras.layers.Conv1D(1, 1, dilation_rate=1)
        
        self.block_1 = ResBlock(32,11,1)
        self.block_2 = ResBlock(32,11,1)
        self.block_3 = ResBlock(32,11,1)
        self.block_4 = ResBlock(32,11,1)
        
        self.crop = tf.keras.layers.Cropping1D(cropping=(40,40))

    def call(self, inputs):
        x_1 = self.conv_1(inputs)
        
        # main branch
        x = self.block_1(x_1)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.conv_3(x)
        
        # residual branch
        x_1 = self.conv_2(x_1)
        
        # come together
        x = x + x_1
        x = self.crop(x)
        x = self.conv_4(x)
        out = tf.keras.activations.sigmoid(x)
        
        return out
    
    
class SpliceAI400(tf.keras.Model):

    def __init__(self):
        super(SpliceAI400, self).__init__()
        self.conv_1 = tf.keras.layers.Conv1D(32, 1, dilation_rate=1)
        self.conv_2 = tf.keras.layers.Conv1D(32, 1, dilation_rate=1)
        self.conv_3 = tf.keras.layers.Conv1D(32, 1, dilation_rate=1)
        self.conv_4 = tf.keras.layers.Conv1D(32, 1, dilation_rate=1)
        self.conv_5 = tf.keras.layers.Conv1D(1, 1, dilation_rate=1)
        
        # first blocks
        self.block_1 = ResBlock(32,11,1)
        self.block_2 = ResBlock(32,11,1)
        self.block_3 = ResBlock(32,11,1)
        self.block_4 = ResBlock(32,11,1)
        
        # second blocks
        self.block_5 = ResBlock(32,11,4)
        self.block_6 = ResBlock(32,11,4)
        self.block_7 = ResBlock(32,11,4)
        self.block_8 = ResBlock(32,11,4)
        
        self.crop = tf.keras.layers.Cropping1D(cropping=(198,199))

    def call(self, inputs):
        x = self.conv_1(inputs)
        x_1 = self.conv_2(x)
        
        # main branch
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x_2 = self.conv_3(x)
        
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.conv_4(x)
        
        # come together
        x = x + x_1 + x_2
        x = self.crop(x)
        x = self.conv_5(x)
        out = tf.keras.activations.sigmoid(x)
        
        return out
        