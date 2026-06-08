
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class MultiScaleDilatedResBlock(layers.Layer):
    """
    Multi-Scale Dilated Residual Block.
    
    Uses three parallel dilated convolutions (dilation rates 1, 2, 4)
    to capture features at different receptive field sizes, then
    concatenates and applies a second convolution with a residual shortcut.
    """

    def __init__(self, filters, **kwargs):
        super(MultiScaleDilatedResBlock, self).__init__(**kwargs)
        self.filters = filters

        # Three parallel dilated convolutions
        self.conv_d1 = layers.Conv2D(filters // 3, 3, padding='same', dilation_rate=1)
        self.conv_d2 = layers.Conv2D(filters // 3, 3, padding='same', dilation_rate=2)
        self.conv_d4 = layers.Conv2D(filters // 3, 3, padding='same', dilation_rate=4)

        self.bn1   = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # Second convolution after concat
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2   = layers.BatchNormalization()

        # 1×1 shortcut projection
        self.shortcut_conv = layers.Conv2D(filters, 1, padding='same')

    def call(self, x, training=False):
        shortcut = self.shortcut_conv(x)

        d1 = self.conv_d1(x)
        d2 = self.conv_d2(x)
        d4 = self.conv_d4(x)

        out = layers.Concatenate()([d1, d2, d4])
        out = self.bn1(out, training=training)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        out = layers.Add()([out, shortcut])
        out = layers.ReLU()(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


class DualAttention(layers.Layer):
    """
    Dual Attention Module — Channel Attention + Spatial Attention (CBAM-style).

    Channel Attention: MLP on global avg-pool and global max-pool outputs.
    Spatial Attention: 7×7 conv on per-pixel avg/max across channels.
    """

    def __init__(self, filters, reduction=8, **kwargs):
        super(DualAttention, self).__init__(**kwargs)
        self.filters   = filters
        self.reduction = reduction

        # Channel Attention
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.gmp = layers.GlobalMaxPooling2D(keepdims=True)
        self.fc1 = layers.Dense(filters // reduction, activation='relu')
        self.fc2 = layers.Dense(filters)

        # Spatial Attention
        self.spatial_conv = layers.Conv2D(1, 7, padding='same')

    def call(self, x):
        # --- Channel attention ---
        avg_pool = tf.reshape(self.gap(x), [-1, self.filters])
        max_pool = tf.reshape(self.gmp(x), [-1, self.filters])

        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))

        channel_att = tf.nn.sigmoid(avg_out + max_out)
        channel_att = tf.reshape(channel_att, [-1, 1, 1, self.filters])
        x = x * channel_att

        # --- Spatial attention ---
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x,  axis=-1, keepdims=True)
        concat  = layers.Concatenate()([avg_out, max_out])

        spatial_att = tf.nn.sigmoid(self.spatial_conv(concat))
        x = x * spatial_att
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'reduction': self.reduction})
        return config


class HiCSuperNetGenerator(Model):
    """
    HiC-SuperNet Generator.

    Architecture:
        - Initial 7×7 feature extraction conv
        - N × MultiScaleDilatedResBlock, with DualAttention inserted every 2 blocks
        - Global residual (skip) connection from initial features
        - 3-stage reconstruction convs (×2, ×1, ×0.5 filters)
        - Linear output conv

    Args:
        base_filters (int): Number of filters in the first conv (default 64).
        num_blocks   (int): Number of MSD residual blocks (default 8).
    """

    def __init__(self, base_filters=64, num_blocks=8, **kwargs):
        super(HiCSuperNetGenerator, self).__init__(**kwargs)
        self.base_filters = base_filters
        self.num_blocks   = num_blocks

        # Initial feature extraction
        self.initial_conv  = layers.Conv2D(base_filters, 7, padding='same',
                                           kernel_initializer='he_normal')
        self.initial_bn    = layers.BatchNormalization()
        self.initial_relu  = layers.ReLU()

        # Stack of MSD Residual Blocks + Dual Attention every 2 blocks
        self.msd_blocks     = []
        self.dual_attentions = []

        for i in range(num_blocks):
            self.msd_blocks.append(
                MultiScaleDilatedResBlock(base_filters, name=f'msd_block_{i}')
            )
            if (i + 1) % 2 == 0:
                self.dual_attentions.append(
                    DualAttention(base_filters, name=f'dual_att_{i}')
                )
            else:
                self.dual_attentions.append(None)

        # Reconstruction
        self.recon_conv1 = layers.Conv2D(base_filters * 2, 3, padding='same',
                                         activation='relu', kernel_initializer='he_normal')
        self.recon_conv2 = layers.Conv2D(base_filters,     3, padding='same',
                                         activation='relu', kernel_initializer='he_normal')
        self.recon_conv3 = layers.Conv2D(base_filters // 2, 3, padding='same',
                                         activation='relu', kernel_initializer='he_normal')

        # Output
        self.output_conv = layers.Conv2D(1, 3, padding='same', activation='linear',
                                         kernel_initializer='he_normal')

    def call(self, x, training=False):
        x = self.initial_conv(x)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)

        initial_features = x   # global skip

        for i in range(self.num_blocks):
            x = self.msd_blocks[i](x, training=training)
            if self.dual_attentions[i] is not None:
                x = self.dual_attentions[i](x)

        x = layers.Add()([x, initial_features])

        x = self.recon_conv1(x)
        x = self.recon_conv2(x)
        x = self.recon_conv3(x)

        return self.output_conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({'base_filters': self.base_filters,
                       'num_blocks':   self.num_blocks})
        return config


def build_model(input_shape=(40, 40, 1), base_filters=64, num_blocks=8, verbose=True):
    """
    Convenience function to instantiate and build HiCSuperNetGenerator.

    Args:
        input_shape  (tuple): Shape of a single input sample (H, W, C).
        base_filters (int):   Base filter count.
        num_blocks   (int):   Number of MSD residual blocks.
        verbose      (bool):  Print model summary if True.

    Returns:
        model: Built HiCSuperNetGenerator instance.
    """
    model = HiCSuperNetGenerator(base_filters=base_filters, num_blocks=num_blocks)
    model.build((None,) + input_shape)

    if verbose:
        print(f"\n{'='*60}")
        print("HiC-SuperNet Architecture Summary")
        print(f"{'='*60}")
        print(f"  Input shape  : {input_shape}")
        print(f"  Base filters : {base_filters}")
        print(f"  MSD blocks   : {num_blocks}")
        print(f"  Total params : {model.count_params():,}")
        print(f"{'='*60}\n")

    return model
