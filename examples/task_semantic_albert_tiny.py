#!/usr/bin/python3
# coding: utf-8

# 语义相似性判断

import json
import numpy as np
import random
import pandas as pd
from random import choice
import unicodedata
import itertools
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
from keras.utils import to_categorical

import sys
sys.path.append('.')

from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.utils import SimpleTokenizer, load_vocab

from config import config_path, checkpoint_path, dict_path, model_save_path, log_dir

INPUT_LENGTH = 128

TEST_DATA_FILE = "./data/test.txt"
TRAIN_DATA_FILE = "./data/train.txt"
DEV_DATA_FILE = "./data/dev.txt"

set_gelu('tanh') # 切换gelu版本

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def read_datas(file_name):
    with open(file_name)as f:
        line = f.readline()
        while line:
            sentence1, sentence2, gold_label = line.strip().split('\t', maxsplit=2)

            yield sentence1, sentence2, int(gold_label)
            line = f.readline()
            line = line.strip()


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class SemanticModel():
    def __init__(self, batch_size=32, train=False):
        self.batch_size = batch_size
        if train:
            chars = set()
            train_datas = read_datas(TRAIN_DATA_FILE)
            dev_datas = read_datas(DEV_DATA_FILE)
            test_datas = read_datas(TEST_DATA_FILE)
            for text1, text2, label in itertools.chain(train_datas, dev_datas):
                chars.update(set(text1))
                chars.update(set(text2))

            _token_dict = load_vocab(dict_path)  # 读取词典
            self.token_dict, self.keep_words = {}, []

            for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
                self.token_dict[c] = len(self.token_dict)
                self.keep_words.append(_token_dict[c])

            for c in chars:
                if c in _token_dict:
                    self.token_dict[c] = len(self.token_dict)
                    self.keep_words.append(_token_dict[c])

            self.tokenizer = SimpleTokenizer(self.token_dict)  # 建立分词器

            with open(os.path.join(model_save_path, 'tokenizer.pkl'), "wb") as f:
                pickle.dump(self.tokenizer, f)

            with open(os.path.join(model_save_path, 'keep_words.pkl'), "wb") as f:
                pickle.dump(self.keep_words, f)

        else:
            with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
                self.tokenizer = pickle.load(f)

            with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
                self.keep_words = pickle.load(f)

        self.model = self.make_model()

    def make_model(self):
        model = load_pretrained_model(
            config_path,
            checkpoint_path,
            keep_words=self.keep_words,
            albert=True
        )

        output = Lambda(lambda x: x[:, 0])(model.output)
        # print(output.shape)
        output = Dense(1, activation='sigmoid')(output) # tanh, sigmoid, softmax
        model = Model(inputs=model.input, outputs=output)

        model.compile(
            loss='binary_crossentropy', # categorical_crossentropy binary_crossentropy
            optimizer=Adam(2e-6),  # 用足够小的学习率
            # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
            metrics=['accuracy']
        )
        model.summary()
        return model

    def gnerator_data(self, file_name):

        X1, X2, Y = [], [], []
        while True:
            for text1, text2, label in read_datas(file_name):

                text1 = text1[:INPUT_LENGTH]
                text2 = text2[:INPUT_LENGTH]
                text1 = unicodedata.normalize('NFKD', text1).strip().lower()
                text2 = unicodedata.normalize('NFKD', text2).strip().lower()
                x1, x2 = self.tokenizer.encode(first=text1, second=text2)
                y = int(label)

                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                # Y.append(to_categorical(y))
                if len(X1) == self.batch_size:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    # print(X1.shape, X2.shape, Y.shape)
                    yield [X1, X2], Y
                    X1, X2, Y = [], [], []

    def train(self):


        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                                 'similarity-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5'),
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

        hist = self.model.fit_generator(
            self.gnerator_data(TRAIN_DATA_FILE),
            steps_per_epoch=1000,
            epochs=100,
            validation_data=self.gnerator_data(DEV_DATA_FILE),
            validation_steps=100,
            callbacks=[early_stopping, model_checkpoint, tb]
        )
        print(hist.history.items())

    def predict(self, text1, text2, weitht_file='similarity-01-0.55-0.741.hdf5'):

        self.model.load_weights(os.path.join(model_save_path, weitht_file), by_name=True,
                           skip_mismatch=True, reshape=True)

        text1 = text1[:INPUT_LENGTH]
        text2 = text2[:INPUT_LENGTH]
        text1 = unicodedata.normalize('NFKD', text1).strip().lower()
        text2 = unicodedata.normalize('NFKD', text2).strip().lower()
        x1, x2 = self.tokenizer.encode(first=text1, second=text2)

        X1 = seq_padding([x1])
        X2 = seq_padding([x2])
        ret = self.model.predict([X1, X2])
        return ret

    def batch_predict(self, question, database):
        text1 = question
        text1 = text1[:INPUT_LENGTH]
        X1, X2 = [], []
        for text2 in database:
            text2 = text2[:INPUT_LENGTH]
            text1 = unicodedata.normalize('NFKD', text1).strip().lower()
            text2 = unicodedata.normalize('NFKD', text2).strip().lower()
            x1, x2 = self.tokenizer.encode(first=text1, second=text2)
            X1.append(x1)
            X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        ret = self.model.predict([X1, X2])

        return ret


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'train':
        semantic_model = SemanticModel(train=True, batch_size=32)
        semantic_model.train()
    else:
        semantic_model = SemanticModel(train=False)
        ret = semantic_model.predict(text1='第一次去见女朋友父母该如何表现？', text2='第一次去见家长该怎么做', weitht_file='similarity-01-0.56-0.736.hdf5')
        print(ret)

if __name__ == '__main__':
    main()


