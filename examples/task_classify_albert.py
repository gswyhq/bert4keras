#! -*- coding:utf-8 -*-
# 意图分类，主题分类

import json
import copy
import numpy as np
import random
import itertools
import pandas as pd
from random import choice
import re, os
import math
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
from keras.utils import plot_model
import sys
sys.path.append('.')

from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.utils import SimpleTokenizer, load_vocab
# from bert4keras.train import PiecewiseLinearLearningRate
from bert4keras.layers import FactorizedEmbedding
from config import albert_model_path, config_path, checkpoint_path, dict_path, model_save_path, log_dir, TERM_FREQUENCY_FILE
set_gelu('tanh') # 切换gelu版本

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

maxlen = 128
model_save_path = './models_classify_albert'

TRAIN_DATA_FILE = './data/train3.txt'
DEV_DATA_FILE = './data/dev3.txt'


def load_vocab_word2id():
    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
        keep_words = pickle.load(f)

    with open(os.path.join(model_save_path, 'label2id.pkl'), "rb") as f:
        label2id = pickle.load(f)
    return tokenizer, keep_words, label2id

def random_weight(weight_data, key_list=None, total=None):
    """
    根据字符串的权重，随机选择
    :param weight_data: {'a': 10, 'b': 2, 'c': 1}
    :param key_list: 随机选择的顺序表
    :param total: 所有权重和
    :return:
    """
    if total is None:
        total = sum(weight_data.values())    # 权重求和

    ra = random.uniform(0, total)   # 在0与权重和之前获取一个随机数
    curr_sum = 0
    ret = None
    if key_list is None:
        key_list = list(weight_data.keys())
        key_list.sort()
    for k in key_list:
        curr_sum += weight_data[k]             # 在遍历中，累加当前权重值
        if ra <= curr_sum:          # 当随机数<=当前权重和时，返回权重key
            ret = k
            break
    return ret

class ProcessData():

    def __init__(self, train=False):
        self.train_count = 0
        self.dev_count = 0
        if train:
            with open(TERM_FREQUENCY_FILE)as f:
                self.words_weight_data = json.load(f)
            self.words_key_list = list(self.words_weight_data.keys())
            self.words_total = sum(self.words_weight_data.values())

    def data_augmentation(self, text):
        """
        数据增强，在句首，句中，句末随机插入一些高频词
        :param text:
        :return:
        """
        word = random_weight(self.words_weight_data, key_list=self.words_key_list, total=self.words_total)
        random_num = random.random()
        if random_num < 0.33:
            text = word + text
        elif random_num < 0.66:
            text = text + word
        else:
            o_index = random.choice([_index for _index, w in enumerate(text)])
            text = text[:o_index] + word + text[o_index:]
        return text

    def generator_process_data(self, data_file='./data/classify_data.txt', data_type='', train=False):
        if train:
            temp_data_file = '{}.temp'.format(data_file)
            print('生成临时文件：{}'.format(temp_data_file))
            command = 'shuf {} -o {}'.format(data_file, temp_data_file)
            os.system(command)
        else:
            print('处理文件：{}'.format(data_file))
            temp_data_file = data_file
        class_weight_count = {}
        with open(temp_data_file, encoding='utf-8')as f:
            line = f.readline()
            while line:
                line = line.strip()
                if not line:
                    line = f.readline()
                    continue
                data = line.rsplit(maxsplit=1)
                if len(data) != 2:
                    line = f.readline()
                    continue
                text, label = data
                if train:
                    # 有些类别的数据太多(最多类别记录有： 2476218， 最小类别记录仅454)，当数据量太多(超过最小标签类别数的100倍)是就按一定概率进行忽略；
                    class_weight_count.setdefault(label, 0)
                    if random.random() < 0.8 and (class_weight_count.get(label, 0) + 1) / (
                            min(class_weight_count.values()) + 1) > 10:
                        line = f.readline()
                        continue
                    else:
                        class_weight_count[label] += 1

                    if data_type == 'train' and random.random() > 0.2 and len(text) < 30:
                        text = self.data_augmentation(text)

                if data_type == 'train':
                    self.train_count += 1
                elif data_type == 'dev':
                    self.dev_count += 1

                yield text, label
                line = f.readline()

        if train:
            os.system('rm {}'.format(temp_data_file))

    def save_word2id_etc(self, datas, incremental_train=False):

        label_set = set()

        _token_dict = load_vocab(dict_path) # 读取词典
        # token_dict, keep_words = {}, set()
        token_dict = copy.deepcopy(_token_dict)
        # for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
        #     token_dict[c] = len(token_dict)
        #     keep_words.add(_token_dict[c])

        for chars, label in datas:
            label_set.add(label)
            # for c in chars:
            #     if c in _token_dict:
            #         token_dict[c] = len(token_dict)
            #         keep_words.add(_token_dict[c])

        # keep_words.add(max(keep_words)+1)
        # keep_words = list(keep_words)
        keep_words = list(set(token_dict.values()))

        tokenizer = SimpleTokenizer(token_dict) # 建立分词器
        label2id = {lab: i for i, lab in enumerate(list(label_set))}

        if not incremental_train:
            with open(os.path.join(model_save_path, 'tokenizer.pkl'), "wb") as f:
                pickle.dump(tokenizer, f)

            with open(os.path.join(model_save_path, 'keep_words.pkl'), "wb") as f:
                pickle.dump(keep_words, f)

            with open(os.path.join(model_save_path, 'label2id.pkl'), "wb") as f:
                pickle.dump(label2id, f)

        return tokenizer, keep_words, label2id


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])



class data_generator:
    def __init__(self, data_file, data_count, batch_size=32, tokenizer=None, label2id=None, data_type='', train=False):
        self.data_file = data_file
        self.batch_size = batch_size
        self.steps = math.ceil(data_count/self.batch_size)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.data_type = data_type
        self.process_data = ProcessData(train=train)
        self.train = train

    def __len__(self):
        return self.steps
        # return 200

    def __iter__(self):
        X1, X2, Y = [], [], []
        while True:
            for text, label in self.process_data.generator_process_data(self.data_file, data_type=self.data_type, train=self.train):
                text = text[:maxlen]
                x1, x2 = self.tokenizer.encode(first=text)
                y = self.label2id.get(label)
                # print("text: {}, y:{},label: {}".format(text, y, label))
                y = to_categorical(y, num_classes=len(self.label2id))
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

class Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        print(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x, **kwargs):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        # print("WQ.shape",WQ.shape)
        # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5)
        QK = K.softmax(QK)
        # print("QK.shape",QK.shape)
        V = K.batch_dot(QK,WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

def make_model(keep_words, label2id, train=False):
    model = load_pretrained_model(
        config_path,
        checkpoint_path,
        keep_words=keep_words,
        albert=True
    )
    class_num = len(label2id)
    # output = Attention(512, name='attention_1')(model.output)
    output = Lambda(lambda x: x[:, 0])(model.output)
    output = Dense(class_num, activation='softmax')(output)
    model = Model(model.input, output)

    # if train:
    #     # 微调albert的顶部几层;
    #     model.trainable = True
    #     set_trainable = False
    #     for layer in model.layers:
    #         if layer.name == 'Encoder-1-FeedForward-Norm': # 'attention_1':
    #             set_trainable = True
    #         if set_trainable:
    #             layer.trainable = True
    #         else:
    #             layer.trainable = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
        metrics=['accuracy']
    )

    # 保存模型图
    plot_model(model, 'classify-albert.png')

    model.summary()
    return model

def train():
    print('开始预处理数据')
    process_data = ProcessData(train=False)
    tokenizer, keep_words, label2id = process_data.save_word2id_etc(
        itertools.chain(process_data.generator_process_data(TRAIN_DATA_FILE, data_type='train'),
                        process_data.generator_process_data(DEV_DATA_FILE, data_type='dev')
                        ),
        incremental_train=False)
    train_count = process_data.train_count
    dev_count = process_data.dev_count

    print('训练集数量： {}；开发集数量：{}'.format(train_count, dev_count))
    print('关系数：{}'.format(len(label2id)))
    model = make_model(keep_words, label2id, train=True)

    train_D = data_generator(TRAIN_DATA_FILE, train_count, tokenizer=tokenizer, label2id=label2id, data_type='train', train=True)
    valid_D = data_generator(DEV_DATA_FILE, dev_count, tokenizer=tokenizer, label2id=label2id, data_type='dev', train=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'classify-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5'),
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

    print('开始训练...')
    model.fit_generator(
        train_D.__iter__(),
        # steps_per_epoch=len(train_D),
        steps_per_epoch = 300,
        epochs=100,
        validation_data=valid_D.__iter__(),
        # validation_steps=len(valid_D),
        validation_steps = 32,
        class_weight='auto',  # 设置class weight让每类的sample对损失的贡献相等。
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint, tb]
    )

def incremental_train(filepath='classify-02-0.62-0.831.hdf5'):
    """
    加载模型增量训练
    :param filepath:
    :return:
    """
    print('开始增量训练...')
    tokenizer, keep_words, label2id = load_vocab_word2id()
    train_count = 1106809
    dev_count = 138236

    print('训练集数量： {}；开发集数量：{}'.format(train_count, dev_count))

    model = make_model(keep_words, label2id, train=True)
    model.load_weights(os.path.join(model_save_path, filepath))

    train_D = data_generator(TRAIN_DATA_FILE, train_count, tokenizer=tokenizer, label2id=label2id, data_type='train', train=True)
    valid_D = data_generator(DEV_DATA_FILE, dev_count, tokenizer=tokenizer, label2id=label2id, data_type='dev', train=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'classify-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5'),
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
        # steps_per_epoch=len(train_D),
        steps_per_epoch = 300,
        initial_epoch = 28,
        epochs=50,
        validation_data=valid_D.__iter__(),
        # validation_steps=len(valid_D),
        validation_steps = 32,
        class_weight='auto',  # 设置class weight让每类的sample对损失的贡献相等。
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint, tb]
    )

def predict(text):
    tokenizer, keep_words, label2id = load_vocab_word2id()

    id2label = {v: k for k, v in label2id.items()}
    # print('id2label={}'.format(id2label))
    model = make_model(keep_words, label2id)
    model.load_weights(os.path.join(model_save_path, 'classify-25-0.41-0.868.hdf5'), by_name=True,
                     skip_mismatch=True, reshape=True)

    text = text[:maxlen]
    x1, x2 = tokenizer.encode(first=text)

    X1 = seq_padding([x1])
    X2 = seq_padding([x2])
    rets = model.predict([X1, X2])
    ret = rets[0]
    # print(ret)
    # leabel = id2label.get(ret.argmax())
    sort_index = ret.argsort()
    leabel = [[id2label.get(_index), float(ret[_index])] for _index in sort_index[-3:][::-1]]
    return leabel

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'incremental_train':
        incremental_train(filepath='classify-25-0.41-0.868.hdf5')
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    else:
        ret = predict(sys.argv[1])
        print(ret)

if __name__ == '__main__':
    main()
