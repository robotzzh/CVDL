import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Input, LayerNormalization, Reshape
from keras.layers import MultiHeadAttention
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras

from keras.callbacks import TensorBoard

# 创建 TensorBoard 回调
log_dir = "./log2s"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# 定义 Transformer 编码层
class TransformerEncoder(keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([
            Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)  # 自注意力
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # 残差连接 + 归一化

        ffn_output = self.ffn(out1)  # 前馈神经网络
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差连接 + 归一化

        return out2


# 加载和预处理 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis].astype("float32") / 255.0
x_test = x_test[..., np.newaxis].astype("float32") / 255.0

# 转换标签为 one-hot 编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 定义输入形状和模型参数
input_shape = x_train.shape[1:]
num_heads = 4
d_model = 64
dff = 128
dropout_rate = 0.1
num_labels = 10

# 构建模型
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Reshape((28*28, 1)))  # 将图像展平成一维并增加一个维度
model.add(Dense(d_model))  # 映射到 d_model 维度

# 添加 Transformer 编码层
model.add(TransformerEncoder(num_heads=num_heads, d_model=d_model, dff=dff, dropout_rate=dropout_rate))

# 添加分类层
model.add(Flatten())  # 展平序列维度
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=320, callbacks=[tensorboard_callback])

# 评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
