#! -*- coding:utf-8 -*-
# 语义相似性判断

import json
import numpy as np
import random
import pandas as pd
from random import choice
import re, os
import codecs
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

import sys
sys.path.append('.')

from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.utils import SimpleTokenizer, load_vocab
# from bert4keras.train import PiecewiseLinearLearningRate
from bert4keras.layers import FactorizedEmbedding
from config import albert_model_path, config_path, checkpoint_path, dict_path, model_save_path, log_dir
set_gelu('tanh') # 切换gelu版本

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

maxlen = 100

TRAIN_DATA_PATH = "/home/gswyhq/data/LCQMC/train.txt"
DEV_DATA_PATH = "/home/gswyhq/data/LCQMC/dev.txt"
TEST_DATA_PATH = "/home/gswyhq/data/LCQMC/test.txt"

def read_data(file_name):
    with open(file_name, encoding='utf-8')as f:
        datas = f.readlines()

    datas = [t.split() for t in datas if t]
    datas = [[t[0], t[1], int(t[2])] for t in datas if len(t) == 3 and t[-1] in ['0', '1']]
    random.shuffle(datas)
    return datas

def process_data(train_file, dev_file, test_file):
    chars = set()

    train_datas = read_data(train_file)
    dev_datas = read_data(dev_file)
    test_datas = read_data(test_file)
    for text1, text2, label in train_datas + dev_datas:
        chars.update(set(text1))
        chars.update(set(text2))

    _token_dict = load_vocab(dict_path) # 读取词典
    token_dict, keep_words = {}, []

    for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

    for c in chars:
        if c in _token_dict:
            token_dict[c] = len(token_dict)
            keep_words.append(_token_dict[c])

    tokenizer = SimpleTokenizer(token_dict) # 建立分词器

    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "wb") as f:
        pickle.dump(tokenizer, f)

    with open(os.path.join(model_save_path, 'keep_words.pkl'), "wb") as f:
        pickle.dump(keep_words, f)

    return train_datas, dev_datas, test_datas, tokenizer, keep_words

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32, tokenizer=None):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.tokenizer = tokenizer
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = np.array([i for i in range(len(self.data))])
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0][:maxlen]
                text2 = d[1][:maxlen]
                x1, x2 = self.tokenizer.encode(first=text1+text2)
                y = d[2]

                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    X1, X2, Y = [], [], []

def make_model(keep_words):
    model = load_pretrained_model(
        config_path,
        checkpoint_path,
        keep_words=keep_words,
        albert=True
    )

    output = Lambda(lambda x: x[:, 0])(model.output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=model.input, outputs=output)

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
        metrics=['accuracy']
    )
    model.summary()
    return model

def train(train_data, valid_data, tokenizer, keep_words):

    model = make_model(keep_words)
    train_D = data_generator(train_data, tokenizer=tokenizer)
    valid_D = data_generator(valid_data, tokenizer=tokenizer)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'similarity-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.3f}.hdf5'),
                                       save_best_only=True, save_weights_only=False)

    tb = TensorBoard(log_dir=log_dir,  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)


    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=2,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[early_stopping, model_checkpoint, tb]
    )

def predict(text1, text2):
    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
        keep_words = pickle.load(f)

    model = make_model(keep_words)
    model.load_weights(os.path.join(model_save_path, 'checkpoint-02-0.15-0.939.hdf5'), by_name=True,
                     skip_mismatch=True, reshape=True)

    text1 = text1[:maxlen]
    text2 = text2[:maxlen]
    x1, x2 = tokenizer.encode(first=text1+text2)

    X1 = seq_padding([x1])
    X2 = seq_padding([x2])
    ret = model.predict([X1, X2])
    return ret

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_data, valid_data, test_data, tokenizer, keep_words = process_data(train_file=TRAIN_DATA_PATH, dev_file=DEV_DATA_PATH, test_file=TEST_DATA_PATH)
        train(train_data[:2000], valid_data[:50], tokenizer, keep_words)
    else:
        ret = predict('这件衣服真漂亮', '这件衣服真好看')
        # 0： 不相似； 1： 相似
        print(ret)

if __name__ == '__main__':
    main()
