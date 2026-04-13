import keras
from keras.layers import Dense, Input, Flatten
from keras.models import Model


class TemporalBlock(keras.layers.Layer):
    """TCN 时间块（因果卷积 + 残差连接）"""

    def __init__(self, n_outputs, kernel_size, dilation_rate, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.n_outputs     = n_outputs
        self.kernel_size   = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate  = dropout

    def build(self, input_shape):
        self.conv1     = keras.layers.Conv1D(self.n_outputs, self.kernel_size,
                                             padding='causal', dilation_rate=self.dilation_rate,
                                             activation='relu')
        self.dropout1  = keras.layers.Dropout(self.dropout_rate)
        self.conv2     = keras.layers.Conv1D(self.n_outputs, self.kernel_size,
                                             padding='causal', dilation_rate=self.dilation_rate,
                                             activation='relu')
        self.dropout2  = keras.layers.Dropout(self.dropout_rate)
        self.downsample = (keras.layers.Conv1D(self.n_outputs, kernel_size=1)
                           if input_shape[-1] != self.n_outputs else None)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        residual = self.downsample(inputs) if self.downsample is not None else inputs
        return keras.activations.relu(x + residual)


def build_tcn_model(input_shape, verbose=True):
    """搭建 TCN 模型并编译"""
    if verbose:
        print(f"   输入形状: {input_shape}")

    inputs = Input(shape=input_shape)
    x = TemporalBlock(32, 3, 1, 0.2)(inputs)
    if verbose:
        print(f"   块1后形状: {x.shape}")
    x = TemporalBlock(32, 3, 2, 0.2)(x)
    if verbose:
        print(f"   块2后形状: {x.shape}")
    x = Flatten()(x)
    if verbose:
        print(f"   展平后形状: {x.shape}")
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name='TCN')
    model.compile(optimizer='adam', loss='mse')
    return model
