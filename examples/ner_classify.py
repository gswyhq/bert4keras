#!/usr/bin/python3
# coding: utf-8

# keras构建两种特征输入，两个输出同时训练
# https://blog.csdn.net/xiaoxiao133/article/details/79653954

import os
import sys
import json
import pickle
import math
import copy
import re
import itertools
import numpy as np
import random
import jieba
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.layers import Embedding, Bidirectional, LSTM, Lambda
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Input, Embedding, LSTM, Dense, Layer
import keras.backend as K
from keras.models import Model
import keras
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
sys.path.append('.')
from config import model_save_path, log_dir, TERM_FREQUENCY_FILE

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" # "0,1"

tf_config = tf.ConfigProto()
tf_config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

model_save_path = './models_ner_classify'
TRAIN_DATA_PATH = "./data/ner_rel_train.txt"
DEV_DATA_PATH = "./data/ner_rel_dev.txt"
TEST_DATA_PATH = "./data/ner_rel_test.txt"

EPOCHS = 100


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

def generator_bio_format(text, entity_dict):
    """
    根据输入的文本及实体词典，生成对应的bio格式数据
    :param text:
    :param entity_dict:
    :return:
    """
    word_flag = []
    if entity_dict and any(key in text for key in entity_dict.keys()):
        index_list = [[i.start(), i.end()] for i in re.finditer('|'.join(list(entity_dict.keys())), text)]
        for _index, char in enumerate(text):
            if any(start_index < _index < end_index for start_index, end_index in index_list):
                continue

            word = ''
            for start_index, end_index in index_list:
                if _index == start_index:
                    word = text[start_index: end_index]
                    break
                elif _index < start_index:
                    break
                else:
                    continue

            if word:
                word = word.upper()
                word_len = len(word)
                word_pinyin = entity_dict.get(word, 'Shiyi').capitalize()
                if word_len == 1:
                    word_flag.append([word, 'B-{}'.format(word_pinyin)])
                elif word_len >= 2:
                    word_flag.append([word[0], 'B-{}'.format(word_pinyin)])
                    for w in word[1:]:
                        word_flag.append([w, 'I-{}'.format(word_pinyin)])
                else:
                    continue
            else:
                word_flag.append([char, 'O'])
    else:
        [word_flag.append([char, 'O']) for char in text]
    # print(''.join([word for word, flag in word_flag]), ''.join([flag for word, flag in word_flag]))
    return word_flag

class Data_set:
    def __init__(self):
        self.word2id = {}
        self.flag2id = {}
        self.rel2id = {}
        self.train_count = 0
        self.dev_count = 0

        with open(TERM_FREQUENCY_FILE)as f:
            self.words_weight_data = json.load(f)
        self.words_key_list = list(self.words_weight_data.keys())
        self.words_total = sum(self.words_weight_data.values())

    def load_data(self, data_path):
        print('读取文件：{}'.format(data_path))
        with open(data_path, "rb") as f:
            data = f.read().decode("utf-8")
        process_data = self.processing_data(data)
        print('预处理完成；')
        return process_data

    def generator_load_data(self, data_path, data_type='train'):
        # print('读取文件：{}'.format(data_path))
        with open(data_path, "r") as f:
            text = f.readline()
            while text:
                text = text.strip()
                if '/' in text:
                    text = text.strip()
                    data = [[word.rsplit('/', maxsplit=1) for word in text.rsplit('\t', maxsplit=1)[0].split() if
                          word[1] == '/'], text.rsplit('\t', maxsplit=1)[-1]]
                    if data_type == 'train':
                        self.train_count += 1
                    elif data_type == 'dev':
                        self.dev_count += 1
                    yield data
                text = f.readline()

    def processing_data(self, data):

        data = [[[word.rsplit('/', maxsplit=1) for word in text.rsplit('\t', maxsplit=1)[0].split() if word[1] == '/'], text.strip().rsplit('\t', maxsplit=1)[-1]] for text in data.split('\n') if '/' in text]
        # random.shuffle(data)
        return data

    def load_vocab(self, save_path):
        with open('./data/train_data.json', 'r')as f:
            train_data = json.load(f)

        with open('./data/dev_data.json', 'r')as f:
            dev_data = json.load(f)

        with open(os.path.join(save_path, 'word2id.pkl'), "rb") as f:
            word2id = pickle.load(f)
        with open(os.path.join(save_path, 'flag2id.pkl'), "rb") as f:
            flag2id = pickle.load(f)
        with open(os.path.join(save_path, 'rel2id.pkl'), "rb") as f:
            rel2id = pickle.load(f)
        self.word2id, self.flag2id, self.rel2id = word2id, flag2id, rel2id

        return train_data, dev_data, word2id, flag2id, rel2id

    def save_vocab(self, save_path, process_data):
        flags = set()
        relationships = set()
        chars = set()
        for word_flag, relationship in process_data:
            chars.update(set(word for word, flag in word_flag))
            flags.update(set(flag for word, flag in word_flag))
            relationships.add(relationship)
        word2id = {char: id_ + 1 for id_, char in enumerate(chars)}
        word2id["unk"] = 0
        flag2id = {label: id_ for id_, label in enumerate(sorted(flags, key=lambda x: 0 if x == 'O' else 1))}
        rel2id = {rel: _id for _id, rel in enumerate(relationships)}
        with open(os.path.join(save_path, 'word2id.pkl'), "wb") as f:
            pickle.dump(word2id, f)
        with open(os.path.join(save_path, 'flag2id.pkl'), "wb") as f:
            pickle.dump(flag2id, f)
        with open(os.path.join(save_path, 'rel2id.pkl'), "wb") as f:
            pickle.dump(rel2id, f)
        self.word2id, self.flag2id, self.rel2id = word2id, flag2id, rel2id
        return word2id, flag2id, rel2id

    def generate_data(self, process_data, input_length):
        char_data_sen = [[token[0] for token in i] for i, j in process_data]
        nerel_sen = [[token[1] for token in i] for i, j in process_data]
        sen2id = [[self.word2id.get(char, 0) for char in sen] for sen in char_data_sen]
        relel_sen = [to_categorical(self.rel2id.get(relel, 0), num_classes=len(self.rel2id)) for i, relel in process_data]
        ner_sen2id = [[self.flag2id.get(ner, 0) for ner in sen] for sen in nerel_sen]

        sen_pad = pad_sequences(sen2id, input_length)
        ner_pad = pad_sequences(ner_sen2id, input_length, value=-1)
        ner_pad = np.expand_dims(ner_pad, 2)
        rel_pad = np.array(relel_sen)
        # shape: (200, 200), (200, 200, 1), (200, 42)
        # print("head1: {}, {}, {}".format(sen_pad[0], ner_pad[0], rel_pad[0]))
        return sen_pad, ner_pad, rel_pad

    def data_augmentation(self, word_flag):
        """
        数据增强，在句首，句中，句末随机插入一些高频词; 转换
        :param word_flag:
        :return:
        """
        text = ''.join([t[0] for t in word_flag])
        entity_dict = {word: flag for word, flag in result_to_json(text, [t[1] for t in word_flag])}

        [jieba.add_word(word, freq=2000, tag=flag) for word, flag in entity_dict.items()]
        origin_words = [word for word in jieba.cut(text)]
        current_words = copy.deepcopy(origin_words)

        random_num = random.random()
        if random_num > 0.5:
            # 调换词序
            random.shuffle(current_words)
            word_flag = generator_bio_format(''.join(current_words), entity_dict)
            return word_flag

        if random_num <= 0.1666:
            insert_index = 0
        elif random_num <= 0.3333:
            insert_index = len(origin_words)
        else:
            # 若插入在居中，则随机选择一个插入位
            insert_index = random.randint(0, len(origin_words))
        random_word = random_weight(self.words_weight_data, key_list=self.words_key_list, total=self.words_total)
        insert_text = ''.join(origin_words[:insert_index] + [random_word] + origin_words[insert_index:])
        word_flag = generator_bio_format(insert_text, entity_dict)
        return word_flag

    def batch_generator(self, data_file, batch_size=32, input_length=200, data_type=''):
        batch_text = []
        batch_ner_tag = []
        batch_rel_tag = []
        class_weight_count = {}
        # self.id2flag = {v: k for k, v in self.flag2id.items()}
        while True:
            temp_data_file = '{}.temp'.format(data_file)
            command = 'shuf {} -o {}'.format(data_file, temp_data_file)
            os.system(command)
            with open(temp_data_file, "r") as f:
                text = f.readline()
                while text:
                    text = text.strip()
                    if '/' in text:
                        word_flag, rel_tag = [[word.rsplit('/', maxsplit=1) for word in text.rsplit('\t', maxsplit=1)[0].split() if
                              word[1] == '/'], text.rsplit('\t', maxsplit=1)[-1]]

                        # 80%概率忽略掉30个字符以上的，80%概率忽略掉没有实体词的语料
                        if (random.random() > 0.2 and len(word_flag) > 30) or (random.random() > 0.2 and all(flag == 'O' for word, flag in word_flag[:20])):
                            text = f.readline()
                            continue

                        if data_type == 'train' and random.random() > 0.2 and input_length - len(word_flag) > 20 and \
                                any(word for word, flag in word_flag if flag == 'O'):
                            word_flag = self.data_augmentation(word_flag)

                        word_flag = word_flag[:input_length]

                        # 有些类别的数据太多(最多类别记录有： 2476218， 最小类别记录仅454)，当数据量太多(超过最小标签类别数的100倍)是就按一定概率进行忽略；
                        class_weight_count.setdefault(rel_tag, 0)
                        if random.random() < 0.8 and (class_weight_count.get(rel_tag, 0) + 1) / (min(class_weight_count.values())+1) > 10:
                            text = f.readline()
                            continue
                        else:
                            class_weight_count[rel_tag] += 1

                        # print('word_flag={}'.format(word_flag))

                        batch_text.append([self.word2id.get(word, 0)  for word, flag in word_flag])
                        flag_ids = []
                        for word, flag in word_flag:
                            # 命名实体不准确，将所有的实体类型都统一为shiyi
                            if flag[0] == 'B':
                                flag = 'B-Shiyi'
                            elif flag[0] == 'I':
                                flag = 'I-Shiyi'
                            flag_id = self.flag2id.get(flag, 0)
                            flag_ids.append(flag_id)
                        batch_ner_tag.append(flag_ids)
                        # print('words: {}; flag_ids: {}; rel_tag`{}`'.format(''.join([word for word, flag in word_flag]), [self.id2flag.get(k) for k in flag_ids], rel_tag))
                        batch_rel_tag.append(to_categorical(self.rel2id.get(rel_tag, 0), num_classes=len(self.rel2id)))
                        if len(batch_ner_tag) >= batch_size:
                            batch_sentence = pad_sequences(batch_text, input_length)
                            ner_pad = pad_sequences(batch_ner_tag, input_length, value=-1)
                            batch_flag_tags = np.expand_dims(ner_pad, 2)
                            batch_rel_tags = np.array(batch_rel_tag)
                            # print('shape: {}'.format(batch_rel_tags[0].shape))
                            yield ({'input': batch_sentence}, {'ner_out': batch_flag_tags, 'rel_out': batch_rel_tags})
                            batch_text = []
                            batch_ner_tag = []
                            batch_rel_tag = []
                    text = f.readline()
                if batch_ner_tag:
                    yield ({'input': batch_sentence}, {'ner_out': batch_flag_tags, 'rel_out': batch_rel_tags})

            os.system('rm {}'.format(temp_data_file))


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

def build_model(input_length, input_dim, ner_units=None, rel_units=None, output_dim=200):
    # construct model
    input = Input((input_length,), dtype='int32', name='input')

    x = Embedding(output_dim=output_dim, input_dim=input_dim, input_length=input_length, mask_zero=True)(input)
    lstm_out = Bidirectional(LSTM(100, return_sequences=True))(x)
    dense = Dense(200, activation='relu')(lstm_out)
    ner_out = CRF(ner_units, sparse_target=True, name='ner_out')(dense)
    # dense = Lambda(lambda x: x, output_shape=lambda s: s)(dense)
    # attention_out = Attention(200, name='attention_1')(dense)
    lambda_out = Lambda(lambda x: x[:, 0])(dense)
    rel_out = Dense(rel_units, activation='softmax', name='rel_out')(lambda_out)

    model = Model(inputs=[input], outputs=[ner_out, rel_out])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
    model.compile(optimizer='adam',
                  loss={'ner_out': crf_loss, 'rel_out': 'categorical_crossentropy'},
                  metrics = {'ner_out': crf_accuracy, 'rel_out': 'accuracy'},
                  loss_weights={'ner_out': 0.5, 'rel_out': 0.5}
                  )
    model.summary()

    # 保存模型图
    plot_model(model, 'ner_classify.png')

    return model

def train(batch_size=32, input_length = 200, epochs=EPOCHS):
    pre_data = Data_set()
    word2id, flag2id, rel2id = pre_data.save_vocab(model_save_path,
                                                   itertools.chain(pre_data.generator_load_data(TRAIN_DATA_PATH, data_type='train'),
                                                                   pre_data.generator_load_data(DEV_DATA_PATH, data_type='dev')))

    print("训练集数据量：{}，验证集数据量：{}".format(pre_data.train_count, pre_data.dev_count))
    print('单词数： {}， 实体类别数：{}， 关系数：{}'.format(len(word2id), len(flag2id), len(rel2id)))
    steps_per_epoch = math.ceil(pre_data.train_count / batch_size)
    validation_steps = math.ceil(pre_data.dev_count / batch_size)

    model = build_model(input_length, input_dim=len(word2id), ner_units=len(flag2id), rel_units=len(rel2id), output_dim=200)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'ner-classify-{epoch:02d}-{ner_out_crf_accuracy:.4f}-{rel_out_acc:.4f}.hdf5'),
                                       save_best_only=True, save_weights_only=False)

    tb = TensorBoard(log_dir=log_dir,  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=batch_size,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    hist = model.fit_generator(pre_data.batch_generator(TRAIN_DATA_PATH, batch_size=batch_size, input_length=input_length, data_type='train'),
                               # batch_size=32,
                               epochs=epochs,
                               # verbose=1,
                               steps_per_epoch = 5000,
                               # validation_split=0.1,
                               validation_data=pre_data.batch_generator(DEV_DATA_PATH, batch_size=batch_size, input_length=input_length),
                               validation_steps = 300,
                               shuffle=True,
                               class_weight= {'ner_out': 'auto', 'rel_out': 'auto'},
                               callbacks=[early_stopping, model_checkpoint, tb]
                     )

    print(hist.history.items())

def incremental_train(batch_size=16, input_length = 200, filepath='classify-02-0.62-0.831.hdf5'):
    """
    加载模型增量训练
    :param filepath:
    :return:
    """
    print('开始增量训练...')
    pre_data = Data_set()
    with open(os.path.join(model_save_path, 'word2id.pkl'), "rb") as f:
        word2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'flag2id.pkl'), "rb") as f:
        flag2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
        rel2id = pickle.load(f)
    print('单词数： {}， 实体类别数：{}， 关系数：{}'.format(len(word2id), len(flag2id), len(rel2id)))
    # print('rel2id={}'.format(rel2id))

    pre_data.word2id = word2id
    pre_data.flag2id = flag2id
    pre_data.rel2id = rel2id

    model = build_model(input_length, input_dim=len(word2id), ner_units=len(flag2id), rel_units=len(rel2id), output_dim=200)
    model.load_weights(os.path.join(model_save_path, filepath))

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'ner-classify-{epoch:02d}-{ner_out_crf_accuracy:.4f}-{rel_out_acc:.4f}.hdf5'),
                                       save_best_only=True, save_weights_only=False)

    tb = TensorBoard(log_dir=log_dir,  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=batch_size,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    hist = model.fit_generator(pre_data.batch_generator(TRAIN_DATA_PATH, batch_size=batch_size, input_length=input_length, data_type='train'),
                               # batch_size=32,
                               epochs=50,
                               # verbose=1,
                               initial_epoch=14,
                               steps_per_epoch = 100,
                               # validation_split=0.1,
                               validation_data=pre_data.batch_generator(DEV_DATA_PATH, batch_size=batch_size, input_length=input_length),
                               validation_steps = 20,
                               shuffle=True,
                               class_weight= {'ner_out': 'auto', 'rel_out': 'auto'},
                               callbacks=[early_stopping, model_checkpoint, tb]
                     )

    print(hist.history.items())

def result_to_json(string, tags):
    entity_name = ""
    entity_type = ''
    datas = []
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            datas.append([char, tag[2:]])
        elif tag[0] == "B":
            if entity_name:
                datas.append([entity_name, entity_type])
                entity_name = ''
                entity_type = ''
            entity_name += char
            entity_type = tag[2:]
        elif tag[0] == "I":
            entity_name += char
            entity_type = tag[2:]
        elif tag[0] == "E":
            entity_name += char
            datas.append([entity_name, tag[2:]])
            entity_name = ""
            entity_type = ''
        else:
            if entity_name:
                datas.append([entity_name, entity_type])
            entity_name = ""
            entity_type = ""

    if entity_name:
        datas.append([entity_name, entity_type])
    return datas

def predict(data, input_length=200):
    with open(os.path.join(model_save_path, 'word2id.pkl'), "rb") as f:
        word2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'flag2id.pkl'), "rb") as f:
        flag2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
        rel2id = pickle.load(f)

    id2flag = {v: k for k, v in flag2id.items()}
    id2rel = {v: k.strip() for k, v in rel2id.items()}
    sen2id = [[word2id.get(char, 0) for char in sen] for sen in data]

    sen_pad = pad_sequences(sen2id, input_length)

    model = build_model(input_length, input_dim=len(word2id), ner_units=len(flag2id), rel_units=len(rel2id), output_dim=200)

    model.load_weights(os.path.join(model_save_path, 'ner-classify-03-0.9658-0.9841.hdf5'), by_name=True,
                     skip_mismatch=True, reshape=True)

    ner_labels, rel_labels = model.predict(sen_pad, input_length)
    results = []
    for ner_label, rel_label, text in zip(ner_labels, rel_labels, data):
        ner_label = [id2flag.get(k.argmax()) for k in ner_label[-len(text):]]
        # argsort函数返回的是数组值从小到大的索引值
        sort_index = rel_label.argsort()
        rel_label = [id2rel.get(_index) for _index in sort_index[-3:][::-1]]
        # rel_label = id2rel.get(rel_label.argmax())
        results.append([text, result_to_json(text, ner_label), rel_label])

    return results

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'incremental_train':
        incremental_train(batch_size=32, input_length = 200, filepath='ner-classify-14-0.9396-0.9841.hdf5')
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        print('开始训练模型。。。')
        train(batch_size=32, input_length = 200)
    else:
        text = sys.argv[1]
        ret = predict([text], input_length=200)
        print("`{}`的命名实体及关系识别的结果：{}".format(text, ret))

if __name__ == '__main__':
    main()