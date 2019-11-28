#!/usr/bin/python3
# coding: utf-8

# 意图识别

import os
import sys
import json
import pickle
import math
import unicodedata
import copy
import re
import itertools
import numpy as np
import random
import jieba
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras_contrib.losses import crf_loss
from keras.layers import Embedding, Bidirectional, LSTM, Lambda, Dropout, TimeDistributed
from keras.models import load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Input, Embedding, LSTM, Dense, Layer
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import time
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from gensim.models import KeyedVectors
from keras.optimizers import Adam
from jieba import re_han_cut_all, re_eng
sys.path.append('.')
from config import albert_model_path, config_path, checkpoint_path, dict_path, model_save_path, log_dir, TERM_FREQUENCY_FILE
from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.utils import SimpleTokenizer, load_vocab
from neo4j_search import search_entity

set_gelu('tanh') # 切换gelu版本



# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" # "0,1"

tf_config = tf.ConfigProto()
tf_config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

model_save_path = './intent_models'

# DATA_FILE = './data/标准项目意图训练语料2019_1019_2.txt'
# gswyhq@gswyhq-PC:~/github_projects/bert4keras/data$ less 标准项目意图训练语料2019_1019_2.txt|sort |uniq |shuf -o 标准项目意图训练语料2019_1019_uniq.txt
# gswyhq@gswyhq-PC:~/github_projects/bert4keras/data$ head -n 5000 标准项目意图训练语料2019_1019_uniq.txt > 标准项目意图训练语料2019_1019_dev.txt
# gswyhq@gswyhq-PC:~/github_projects/bert4keras/data$ tail -n 47946 标准项目意图训练语料2019_1019_uniq.txt > 标准项目意图训练语料2019_1019_train.txt

TRAIN_DATA_FILE = './data/标准项目意图训练语料2019_1019_train.txt'
DEV_DATA_FILE = './data/标准项目意图训练语料2019_1019_dev.txt'

EPOCHS = 100

def split_text(text):
    """
    将一个句子转换为一个列表
    若是英文，保留英文单词
    中文切分成单字
    :param text: baoxianchanpin jibing能换成其他的险种吗？
    :return: ['baoxianchanpin', 'jibing', '能', '换', '成', '其', '他', '的', '险', '种', '吗', '？']
    """
    words = []
    for word in re_han_cut_all.split(text):
        if re_eng.match(word):
            words.extend(word.split())
        else:
            words.extend(list(word))
    return words

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

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class Data_set:
    def __init__(self):
        self.keep_words = None
        self.tokenizer = None
        self.word2id = {}
        self.rel2id = {}


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
                text = unicodedata.normalize('NFKD', text).strip()
                if '\t' in text:
                    text, rel = text.split('\t', maxsplit=1)
                    yield text, rel
                text = f.readline()

    def processing_data(self, data):

        data = [line.strip().split('\t', maxsplit=1) for line in data if '\t' in line.strip()]
        random.shuffle(data)
        return data

    def load_vocab(self, save_path):
        with open('./data/train_data.json', 'r')as f:
            train_data = json.load(f)

        with open('./data/dev_data.json', 'r')as f:
            dev_data = json.load(f)

        with open(os.path.join(save_path, 'tokenizer.pkl'), "rb") as f:
            tokenizer = pickle.load(f)
        with open(os.path.join(save_path, 'word2id.pkl'), "rb") as f:
            word2id = pickle.load(f)
        with open(os.path.join(save_path, 'rel2id.pkl'), "rb") as f:
            rel2id = pickle.load(f)
        self.tokenizer, self.word2id, self.rel2id = tokenizer, word2id, rel2id

        return train_data, dev_data, tokenizer, word2id, rel2id

    def save_vocab(self, save_path, process_data):
        chars = set()
        relationships = set()
        for text, relationship in process_data:
            words = split_text(text)
            chars.update(set(words))
            relationships.add(relationship)

        token_dict = load_vocab(dict_path)  # 读取词典
        keep_chars = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']

        for char in chars:
            if not token_dict.get(char):
                # token_dict[char] = len(token_dict)
                keep_chars.append(char)

        # for char in keep_chars:
        #     if not token_dict.get(char):
        #         token_dict[char] = len(token_dict)

        keep_words = list(set(token_dict.values()))

        tokenizer = SimpleTokenizer(token_dict) # 建立分词器


        word2id = {word: id_ + len(keep_chars) for id_, word in enumerate(chars)}
        for _id, word in enumerate(keep_chars):
            word2id[word] = _id

        rel2id = {rel: _id for _id, rel in enumerate(relationships)}

        with open(os.path.join(model_save_path, 'tokenizer.pkl'), "wb") as f:
            pickle.dump(tokenizer, f)

        with open(os.path.join(model_save_path, 'keep_words.pkl'), "wb") as f:
            pickle.dump(keep_words, f)

        with open(os.path.join(save_path, 'word2id.pkl'), "wb") as f:
            pickle.dump(word2id, f)
        with open(os.path.join(save_path, 'rel2id.pkl'), "wb") as f:
            pickle.dump(rel2id, f)

        self.tokenizer, self.word2id, self.rel2id = tokenizer, word2id, rel2id

        return tokenizer, keep_words, word2id, rel2id

    def data_augmentation(self, text):
        """
        数据增强，在句首，句中，句末随机插入一些高频词; 转换
        :param text:
        :return:
        """

        origin_words = [word for word in jieba.cut(text)]
        current_words = copy.deepcopy(origin_words)

        random_num = random.random()

        if random_num > 0.75:
            # 调换词序
            random.shuffle(current_words)
            text = ''.join(current_words)
            return text
        elif random_num > 0.5:
            insert_index = 0
        elif random_num > 0.25:
            insert_index = len(origin_words)
        else:
            # 若插入在居中，则随机选择一个插入位
            insert_index = random.randint(0, len(origin_words))
        random_word = random_weight(self.words_weight_data, key_list=self.words_key_list, total=self.words_total)
        text = ''.join(origin_words[:insert_index] + [random_word] + origin_words[insert_index:])

        return text

    def batch_generator(self, data_file, batch_size=32, input_length=200):
        X1, X2 = [], []
        Y_REL = []
        class_weight_count = {}
        # self.id2word = {v: k for k, v in self.word2id.items()}
        while True:
            temp_data_file = '{}.temp'.format(data_file)
            command = 'ldconfig && less {} | sort |uniq | shuf -o {}'.format(data_file, temp_data_file)
            os.system(command)
            with open(temp_data_file, "r") as f:
                line = f.readline()
                while line:
                    line = unicodedata.normalize('NFKD', line).strip()
                    if '\t' in line:

                        text, rel_tag = line.split('\t')

                        # # 80%概率忽略掉30个字符以上的，80%概率忽略掉没有实体词的语料
                        # if (random.random() > 0.2 and len(word_flag) > 30) or (random.random() > 0.2 and all(flag == 'O' for word, flag in word_flag[:20])):
                        #     line = f.readline()
                        #     continue

                        # word_flag = word_flag[:input_length]

                        # 有些类别的数据太多(最多类别记录有： 2476218， 最小类别记录仅454)，当数据量太多(超过最小标签类别数的100倍)是就按一定概率进行忽略；
                        class_weight_count.setdefault(rel_tag, 0)
                        if random.random() < 0.8 and (class_weight_count.get(rel_tag, 0) + 1) / (min(class_weight_count.values())+1) > 10:
                            line = f.readline()
                            continue
                        else:
                            class_weight_count[rel_tag] += 1

                        if random.random() > 0.8:
                            text = self.data_augmentation(text)
                            # print('数据增强的结果：{}'.format([[word for word, flag in word_flag], [flag for word, flag in word_flag]]))

                        # print('word_flag={}'.format(word_flag))
                        # batch_text.append([self.word2id.get(word, 0)  for word, flag in word_flag])
                        # text = ''.join([word for word, flag in word_flag])
                        text = text[:input_length]
                        x1, x2 = self.tokenizer.encode(text)
                        X1.append(x1)
                        X2.append(x2)

                        # print('words: {}; flag_ids: {}; rel_tag`{}`'.format(''.join([word for word, flag in word_flag]), [self.id2word.get(k) for k in flag_ids], rel_tag))
                        Y_REL.append(to_categorical(self.rel2id.get(rel_tag, 0), num_classes=len(self.rel2id)))
                        if len(X1) >= batch_size:
                            X1 = seq_padding(X1)
                            X2 = seq_padding(X2)
                            Y_REL = seq_padding(Y_REL)
                            yield ({'Input-Token': X1, 'Input-Segment': X2}, {'rel_out': Y_REL})
                            X1, X2 = [], []
                            Y_REL = []
                    line = f.readline()
                if X1:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y_REL = seq_padding(Y_REL)

                    yield ({'Input-Token': X1, 'Input-Segment': X2}, {'rel_out': Y_REL})
                    X1, X2 = [], []
                    Y_REL = []

            os.system('rm {}'.format(temp_data_file))
            time.sleep(3)

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

def get_not_present_word_vectors(low=0.0, high=1.0, size=None, normalize=False):
    """
    随机初始化一个向量
    :param low: 向量最小值
    :param high: 向量元素最大值
    :param size: 向量长度
    :param normalize: 是否正则化
    :return:
    """
    vec = np.random.uniform(low=low, high=high, size=size)
    if normalize:
        # 所有值都除以l2范数（平方和再开方）
        vec /= np.linalg.norm(vec, ord=2)

    return vec

def load_word_embedding_weights(word_embedding_file):
    wv_from_text = KeyedVectors.load_word2vec_format(word_embedding_file, binary=False)
    words = wv_from_text.wv.index2word
    word_embedding_weights = np.array([wv_from_text.get_vector(word) for word in words], dtype='float32', copy=False)
    return word_embedding_weights

def build_model(keep_words, rel_units=None):
    # construct model

    model = load_pretrained_model(
        config_path,
        checkpoint_path,
        keep_words=keep_words,
        # keep_words = None,
        albert=True
    )

    output = Lambda(lambda x: x[:, 0])(model.output)

    # dense = Dense(200, activation='relu')(output)
    # dense = BatchNormalization()(dense)
    # ner_out = CRF(ner_units, sparse_target=True, name='ner_out')(model.output)
    # dense = Lambda(lambda x: x, output_shape=lambda s: s)(dense)
    # attention_out = Attention(200, name='attention_1')(dense)
    # lambda_out = Lambda(lambda x: x[:, 0])(dense)
    # lambda_out = BatchNormalization()(lambda_out)
    rel_out = Dense(rel_units, activation='softmax', name='rel_out')(output)

    model = Model(model.input, outputs=[rel_out])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
    # lr不能太低，如：5e-7，太低了，反而会使模型训练还没达到最优就提前终止了；
    model.compile(optimizer=Adam(lr=5e-6),
                  loss={'rel_out': 'categorical_crossentropy'},
                  metrics = {'rel_out': 'accuracy'},
                  # loss_weights={'rel_out': 0.5}
                  )
    model.summary()

    # 保存模型图
    plot_model(model, 'intent_albert_tiny.png')

    return model

def train(batch_size=32, input_length = 200, epochs=EPOCHS):
    pre_data = Data_set()
    tokenizer, keep_words, word2id, rel2id = pre_data.save_vocab(model_save_path, itertools.chain(pre_data.generator_load_data(TRAIN_DATA_FILE),
                                                                                                  pre_data.generator_load_data(DEV_DATA_FILE)))


    print('单词数：{}， 关系数：{}'.format(len(word2id), len(rel2id)))
    # print('字向量维度：{}'.format(word_embedding_weights.shape))
    # steps_per_epoch = math.ceil(pre_data.train_count / batch_size)
    # validation_steps = math.ceil(pre_data.dev_count / batch_size)

    model = build_model(keep_words, rel_units=len(rel2id))
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'intent-albert-tiny-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.hdf5'),
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

    hist = model.fit_generator(pre_data.batch_generator(TRAIN_DATA_FILE, batch_size=batch_size, input_length=input_length),
                               # batch_size=32,
                               epochs=epochs,
                               # verbose=1,
                               steps_per_epoch = 1000,
                               # validation_split=0.1,
                               validation_data=pre_data.batch_generator(DEV_DATA_FILE, batch_size=batch_size, input_length=input_length),
                               validation_steps = 100,
                               shuffle=True,
                               # class_weight= {'ner_out': 'auto', 'rel_out': 'auto'},
                               callbacks=[early_stopping, model_checkpoint, tb]
                     )

    print(hist.history.items())

def evaluate(test_file, input_length=200, weight_file='intent-albert-tiny-33-0.9731-0.9852.hdf5'):
    with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
        keep_words = pickle.load(f)
    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_save_path, 'word2id.pkl'), "rb") as f:
        word2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
        rel2id = pickle.load(f)
    id2rel = {v: k for k, v in rel2id.items()}
    id2word = {v: k for k, v in word2id.items()}
    print('单词数： {}， 单词数：{}， 关系数：{}'.format(len(keep_words), len(word2id), len(rel2id)))
    # print('rel2id={}'.format(rel2id))

    model = build_model(keep_words, rel_units=len(rel2id))

    model.load_weights(os.path.join(model_save_path, weight_file), by_name=True,
                     skip_mismatch=True, reshape=True)

    X1 = []
    X2 = []
    results = []
    data = []
    ner_count = 0
    rel_count = 0
    with open(test_file)as f:
        for line in f.readlines():
            line = line.strip()
            word_flag, rel_tag = [[word.rsplit('/', maxsplit=1) for word in line.rsplit('\t', maxsplit=1)[0].split() if
                               word[1] == '/'], line.rsplit('\t', maxsplit=1)[-1]]
            data.append([''.join([word for word, flag in word_flag]), [flag for word, flag in word_flag], rel_tag])
            if len(data) >= 32:
                for text, flags, rel_tag in data:
                    text = text[:input_length]
                    x1, x2 = tokenizer.encode(text)
                    X1.append(x1)
                    X2.append(x2)
                X1 = seq_padding(X1)
                X2 = seq_padding(X2)
                rets = model.predict([X1, X2])
                ner_labels, rel_labels = rets
                # print(rets)
                # print([ret.shape for ret in rets])

                for ner_label, rel_label, (text, flags, rel_tag) in zip(ner_labels, rel_labels, data):
                    # print(ner_label.shape)
                    # print(ner_label[-len(text):])
                    ner_label = [id2word.get(k.argmax()) for k in ner_label[1:-1]]
                    # argsort函数返回的是数组值从小到大的索引值

                    rel_label = id2rel.get(rel_label.argmax())
                    # rel_label = id2rel.get(rel_label.argmax())
                    results.append([text, ner_label, rel_label, flags, rel_tag])
                    if all(k[0]==v[0] for k, v in zip(ner_label, flags)):
                        ner_count += 1
                    if rel_label == rel_tag:
                        rel_count += 1

                    if len(results) % 100 == 0:
                        print('已完成测试量：{}'.format(len(results)))

                    if len(results) % 1000 == 0:
                        print('测试集总量：{}\n命名实体识别正确率：{}\n关系属性识别正确率：{}'.format(len(results), ner_count / len(results),
                                                                            rel_count / len(results)))

                X1 = []
                X2 = []
                data = []
                # print('results={}'.format(results))
                # sys.exit(0)

            if len(results) > 50000:
                sys.exit(0)

    print('测试集总量：{}\n命名实体识别正确率：{}\n关系属性识别正确率：{}'.format(len(results), ner_count/len(results), rel_count/len(results)))

    return results

def incremental_train(batch_size=16, input_length = 200, filepath='classify-02-0.62-0.831.hdf5'):
    """
    加载模型增量训练
    :param filepath:
    :return:
    """
    print('开始增量训练...')
    pre_data = Data_set()
    with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
        keep_words = pickle.load(f)
    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_save_path, 'word2id.pkl'), "rb") as f:
        word2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
        rel2id = pickle.load(f)
    print('单词数： {}， 单词数：{}， 关系数：{}'.format(len(keep_words), len(word2id), len(rel2id)))
    # print('rel2id={}'.format(rel2id))

    pre_data.keep_words = keep_words
    pre_data.tokenizer = tokenizer
    pre_data.word2id = word2id
    pre_data.rel2id = rel2id

    model = build_model(keep_words, rel_units=len(rel2id))
    model.load_weights(os.path.join(model_save_path, filepath))

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'intent-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.hdf5'),
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

    hist = model.fit_generator(pre_data.batch_generator(TRAIN_DATA_FILE, batch_size=batch_size, input_length=input_length),
                               # batch_size=32,
                               epochs=50,
                               # verbose=1,
                               initial_epoch=14,
                               steps_per_epoch = 100,
                               # validation_split=0.1,
                               validation_data=pre_data.batch_generator(DEV_DATA_FILE, batch_size=batch_size, input_length=input_length),
                               validation_steps = 20,
                               shuffle=True,
                               class_weight= {'rel_out': 'auto'},
                               callbacks=[early_stopping, model_checkpoint, tb]
                     )

    print(hist.history.items())

def predict(data, input_length=200, weight_file='intent-albert-tiny-33-0.9731-0.9852.hdf5'):
    with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
        keep_words = pickle.load(f)
    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_save_path, 'word2id.pkl'), "rb") as f:
        word2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
        rel2id = pickle.load(f)
    id2rel = {v: k for k, v in rel2id.items()}
    id2word = {v: k for k, v in word2id.items()}
    print('单词数： {}， 单词数：{}， 关系数：{}'.format(len(keep_words), len(word2id), len(rel2id)))
    # print('rel2id={}'.format(rel2id))

    model = build_model(keep_words, rel_units=len(rel2id))

    model.load_weights(os.path.join(model_save_path, weight_file), by_name=True,
                     skip_mismatch=True, reshape=True)

    X1 = []
    X2 = []
    for text in data:
        text = text[:input_length]
        x1, x2 = tokenizer.encode(text)
        X1.append(x1)
        X2.append(x2)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    rel_labels = model.predict([X1, X2])
    # print(rets)
    # print([ret.shape for ret in rets])
    results = []
    for rel_label, text in zip(rel_labels, data):
        # print(ner_label.shape)
        # print(ner_label[-len(text):])
        # print('ner_label={}'.format([list(t) for t in ner_label]))
        # print('id2word={}'.format(id2word))
        # ner_label = [id2word.get(k.argmax()) for k in ner_label[1:-1]]
        # print(ner_label)
        # argsort函数返回的是数组值从小到大的索引值
        sort_index = rel_label.argsort()
        rel_label = [id2rel.get(_index) for _index in sort_index[-3:][::-1]]
        # rel_label = id2rel.get(rel_label.argmax())
        results.append([text, rel_label])

    return results

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'incremental_train':
        incremental_train(batch_size=32, input_length = 200, filepath='intent-14-0.9396-0.9841.hdf5')
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        print('开始训练模型...')
        train(batch_size=32, input_length = 200)
    elif len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        if len(sys.argv) > 2:
            test_file = sys.argv[2]
        else:
            test_file = TEST_DATA_PATH
        evaluate(test_file, input_length=200, weight_file='intent-albert-tiny-34-0.9465-0.9790.hdf5')
    else:
        text = sys.argv[1]
        ret = predict([text], input_length=200, weight_file='intent-albert-tiny-34-0.9835-0.9446.hdf5')
        print("`{}`的命名实体及关系识别的结果：{}".format(text, ret))

if __name__ == '__main__':
    main()
