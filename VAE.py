#! -*- coding:utf-8 -*-


import os
import tensorflow as tf
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from tensorboardX import SummaryWriter
import Data  
root_dir = os.path.dirname(os.path.abspath(__file__))

# 超参数
epochs = 1000        # 训练轮数
batch_size = 128     # 批量大小
hidden_dim = 64      # 隐层节点数
latent_dim = 64     # 隐变量维度
optimizer = Adam(0.001)

# 获取训练数据
data_path = os.path.join(root_dir, "data")
dga, _ = Data.get_data(data_path)             # 

# 获取数据集相关参数
id2char = Data.id2char
max_len = Data.max_len
if(0  in dga):
    vocab_len = Data.MAX_FEATURES + 1 
else:
    vocab_len = Data.MAX_FEATURES         


##########################################################################################################################################################
##########################################################################################################################################################
class GCNN(Layer): # 定义GCNN层，结合残差
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                     shape=(3, input_shape[-1],
                                            self.output_dim * 2),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, x):
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = _[:,:,:self.output_dim] * K.sigmoid(_[:,:,self.output_dim:])
        if self.residual:
            return _ + x
        else:
            return _
    def get_config(self):
        config = super(GCNN, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'residual': self.residual
        })
        return config    

# 定义编码层---Encoder
encoder_input = Input(shape=(max_len,))                         # 
embLayer = Embedding(vocab_len, hidden_dim)(encoder_input)      # (None, max_len, hidden_dim)

GCNNLayer1 = GCNN(residual=True)(embLayer)                      # (None, max_len, hidden_dim)
GCNNLayer2 = GCNN(residual=True)(GCNNLayer1)                    # (None, max_len, hidden_dim)
GAPLayer = GlobalAveragePooling1D()(GCNNLayer2)                 # (None, hidden_dim)

BlstmLayer = Bidirectional(LSTM(hidden_dim))(embLayer)          # (None, hidden_dim*2)
BlstmLayer = Dense(hidden_dim)(BlstmLayer)                      # (None, hidden_dim)

h = concatenate([GAPLayer, BlstmLayer])                         # (None, hidden_dim*2)
h = Dense(512, activation='relu')(GAPLayer)
h = Dropout(0.5)(h)
h = Dense(256, activation='relu')(h)
h = Dropout(0.2)(h)
h = Dense(128, activation='relu')(h)
z_mean = Dense(latent_dim)(h)                   # 均值层    (None, latent_dim)
z_log_var = Dense(latent_dim)(h)                # 方差层    (None, latent_dim)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling)([z_mean, z_log_var])

###########################################################################################################################################
# 定义解码层---Decoder       
decoder_hidden = Dense(hidden_dim*(max_len))            # 隐层    
decoder_cnn = GCNN(residual=True)                       # GCNN层 
decoder_cnn2 = GCNN(residual=True)                      # GCNN层  
decoder_fc1 = Dense(512, activation='relu')             # 全连接层 
decoder_dropout = Dropout(0.5)                          # Dropout层
decoder_fc2 = Dense(256, activation='relu')             # 全连接层
decoder_dense = Dense(vocab_len, activation='softmax')  # 输出层

h = decoder_hidden(z)                   # 隐层 维度为hidden_dim*max_len
h = Reshape((max_len, hidden_dim))(h)   # 转换为维度为(max_len, hidden_dim)
h = decoder_cnn(h)                      # GCNN层 维度为(max_len, hidden_dim)
h = decoder_cnn2(h)                     # GCNN层 维度为(max_len, hidden_dim)
h = decoder_fc1(h)                      # 全连接层 维度为(max_len, 1024)
h = decoder_dropout(h)                  # Dropout层 维度为(max_len, 1024)
h = decoder_fc2(h)                      # 全连接层 维度为(max_len, 512)
decoder_output = decoder_dense(h)       # 输出层 维度为(max_len, vocab_len)
VAE = Model(encoder_input, decoder_output)   

# #############################################################################################################################################################
# xent_loss是重构loss，也就是encoder输入和decoder输出的LOSS, kl_loss是KL loss,Z和X'之间的损失
# 损失函数=输入输出的差异+高斯分布(z_mean, ) kl散度
xent_loss = K.sum(K.sparse_categorical_crossentropy(encoder_input, decoder_output), 1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
VAE.add_loss(vae_loss)
VAE.compile(optimizer=optimizer)
VAE.summary()
#############################################################################################################################################################
# 重用解码层，构建单独的生成模型
decoder_input = Input(shape=(latent_dim,))
_ = decoder_hidden(decoder_input)
_ = Reshape((max_len, hidden_dim))(_)
_ = decoder_cnn(_)
_ = decoder_cnn2(_)
_ = decoder_fc1(_)
_ = decoder_dropout(_)
_ = decoder_fc2(_)
_output = decoder_dense(_)
generator = Model(decoder_input, _output)
#############################################################################################################################################################

file_name = os.path.basename(data_path).split('.')[0]
global ep                    
ep = 0

# 域名转换函数，将域名数字列表转换为域名字符串
def list2domain(idTabel: list):
    s = ''
    for i in idTabel:
        if(i==0):
            s += '+'
        else:
            s+=id2char[i]
    return s

# 生成域名函数
def gen(f):
    res = generator.predict(np.random.randn(1, latent_dim))[0]
    res = res.argmax(axis=1)
    domain = list2domain(res) 
    global ep
    ep += 1
    f.write("Epoch %d gen domain : %s\n"%(ep,domain))
    return domain

log_dir = "./logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir)
# 回调器，方便在训练过程中输出
class Evaluate(Callback):
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.log = []
        self.writer = None
        self.genDomain_path = "./genDomain/Gen.txt"

    def on_train_begin(self, logs=None):
        self.f = open(self.genDomain_path, "a")
        self.writer = SummaryWriter(logdir=self.log_dir,flush_secs=30)

    def on_epoch_end(self, epoch, logs=None):
        self.log.append(gen(self.f))
        print (u'          %s'%(self.log[-1]))      # 输出最新生成的域名
    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            loss = logs.get('loss')
            self.writer.add_scalar('batch_loss', loss, self.model.optimizer.iterations)

    def on_train_end(self, logs=None):
        self.writer.close()
        self.f.close()

evaluator = Evaluate(log_dir)


# 训练VAE模型
VAE.fit(dga,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[evaluator, tensorboard_callback])

VAE.save('./genModel/gen_model.h5')
