#! -*- coding:utf-8 -*-
# 意图分类，主题分类

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
from keras.utils.np_utils import to_categorical
import sys
sys.path.append('.')

from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.utils import Tokenizer, load_vocab
# from bert4keras.train import PiecewiseLinearLearningRate
from bert4keras.layers import FactorizedEmbedding
from config import albert_model_path, config_path, checkpoint_path, dict_path, model_save_path, log_dir
set_gelu('tanh') # 切换gelu版本

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

maxlen = 100
model_save_path = './models_classify_albert'

TRAIN_DEV_DATA = './data/classify_data.txt'

def process_data(data_file='./data/classify_data.txt'):
    with open(data_file, encoding='utf-8')as f:
        datas = f.readlines()

    chars = set()
    labels = set()
    new_datas = []
    for data in datas:
        data = data.strip()
        if not data:
            continue
        text, label = data.rsplit(maxsplit=1)
        chars.update(set(text))
        labels.add(label)
        new_datas.append([text, label])
    del datas

    label2id = {lab: i for i, lab in enumerate(list(labels))}

    _token_dict = load_vocab(dict_path) # 读取词典
    token_dict, keep_words = {}, []

    for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

    for c in chars:
        if c in _token_dict:
            token_dict[c] = len(token_dict)
            keep_words.append(_token_dict[c])


    tokenizer = Tokenizer(token_dict) # 建立分词器

    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "wb") as f:
        pickle.dump(tokenizer, f)

    with open(os.path.join(model_save_path, 'keep_words.pkl'), "wb") as f:
        pickle.dump(keep_words, f)

    with open(os.path.join(model_save_path, 'label2id.pkl'), "wb") as f:
        pickle.dump(label2id, f)

    if not os.path.exists('./random_order.json'):
        random_order = [i for i in range(len(new_datas))]
        random.shuffle(random_order)
        json.dump(
            random_order,
            open('./random_order.json', 'w'),
            indent=4
        )
    else:
        random_order = json.load(open('./random_order.json'))

    # 按照9:1的比例划分训练集和验证集
    train_data = [new_datas[j] for i, j in enumerate(random_order) if i % 10 != 0]
    valid_data = [new_datas[j] for i, j in enumerate(random_order) if i % 10 == 0]

    return train_data, valid_data, tokenizer, keep_words, label2id

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32, tokenizer=None, label2id=None):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.tokenizer = tokenizer
        self.label2id = label2id
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
                text = d[0][:maxlen]
                x1, x2 = self.tokenizer.encode(text)
                y = self.label2id.get(d[1])
                y = to_categorical(y, num_classes=len(self.label2id))
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    # print('X1: {}, X2: {}, Y: {}'.format(X1.shape, X2.shape, Y.shape))
                    # print('X1: {}, X2: {}, Y: {}'.format(X1[0].shape, X2[0].shape, Y[0].shape))
                    # print('退出')
                    # exit(0)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

def make_model(keep_words, label2id):
    model = load_pretrained_model(
        config_path,
        checkpoint_path,
        keep_words=keep_words,
        albert=True
    )
    class_num = len(label2id)
    output = Lambda(lambda x: x[:, 0])(model.output)
    output = Dense(class_num, activation='softmax')(output)
    model = Model(model.input, output)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
        metrics=['accuracy']
    )
    model.summary()
    return model

def train(train_data, valid_data, tokenizer, keep_words, label2id):

    model = make_model(keep_words, label2id)

    train_D = data_generator(train_data, tokenizer=tokenizer, label2id=label2id)
    valid_D = data_generator(valid_data, tokenizer=tokenizer, label2id=label2id)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'checkpoint-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.3f}.hdf5'),
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
        epochs=200,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[early_stopping, model_checkpoint, tb]
    )

def predict(text):
    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
        keep_words = pickle.load(f)

    with open(os.path.join(model_save_path, 'label2id.pkl'), "rb") as f:
        label2id = pickle.load(f)

    id2label = {v: k for k, v in label2id.items()}

    model = make_model(keep_words, label2id)
    model.load_weights(os.path.join(model_save_path, 'checkpoint-02-0.15-0.939.hdf5'), by_name=True,
                     skip_mismatch=True, reshape=True)

    text = text[:maxlen]
    x1, x2 = tokenizer.encode(text)

    X1 = seq_padding([x1])
    X2 = seq_padding([x2])
    ret = model.predict([X1, X2])
    return ret

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_data, valid_data, tokenizer, keep_words, label2id = process_data(data_file=TRAIN_DEV_DATA)
        train(train_data, valid_data, tokenizer, keep_words, label2id)
    else:
        ret = predict('这件衣服真漂亮')
        print(ret)

if __name__ == '__main__':
    main()
