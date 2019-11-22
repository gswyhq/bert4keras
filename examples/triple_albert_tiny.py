#! -*- coding:utf-8 -*-
# 候选三元组分类

import json
import copy
import numpy as np
import random
import itertools
import pandas as pd
from random import choice
import re, os

import unicodedata
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

from examples.ner_classify import result_to_json
from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.utils import SimpleTokenizer, load_vocab
from utils.es_search_three_tuple import search_three_tuple, get_random_chat_question
from config import albert_model_path, config_path, checkpoint_path, dict_path, model_save_path, log_dir, TERM_FREQUENCY_FILE
set_gelu('tanh') # 切换gelu版本

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

maxlen = 128
model_save_path = './models_triple_albert_tiny'

TRAIN_DATA_FILE = './data/ner_rel_train_BIOES.txt.temp.aug.triple'
DEV_DATA_FILE = './data/ner_rel_dev_BIOES.txt.temp.aug.triple'

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


class Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
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
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        # print("QK.shape",QK.shape)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class TripleModel():
    def __init__(self, train=False, train_file=TRAIN_DATA_FILE, dev_file=DEV_DATA_FILE, batch_size=32, input_length=128):
        self.train_file = train_file
        self.dev_file = dev_file
        self.batch_size = batch_size
        self.input_length = input_length

        if train:
            with open(TERM_FREQUENCY_FILE)as f:
                self.words_weight_data = json.load(f)
            self.words_key_list = list(self.words_weight_data.keys())
            self.words_total = sum(self.words_weight_data.values())

            self.tokenizer, self.keep_words, self.rel2id = self.save_vocab(itertools.chain(self.gernerator_data(TRAIN_DATA_FILE),
                                                              self.gernerator_data(DEV_DATA_FILE)))
        else:
            with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
                self.tokenizer = pickle.load(f)

            with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
                self.keep_words = pickle.load(f)

            with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
                self.rel2id = pickle.load(f)

    def gernerator_data(self, input_file):
        with open(input_file) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                word_flag = data['word_flag']
                rel = data['rel']
                triple = data['triple']
                text = ''.join([word for word, flag in word_flag])
                text = unicodedata.normalize('NFKD', text).strip()
                yield [text, triple], [result_to_json(text, [flag for word, flag in word_flag]), rel]

                line = f.readline()

    def save_vocab(self, input_data, incremental_train=False):
        relationships = set()
        chars = set()
        for (text, triple), (entity_lists, rel) in input_data:
            chars.update(set(text))
            relationships.add(rel)
            relationships.update(set(p for s, p, o in triple))

        token_dict = load_vocab(dict_path)  # 读取词典

        keep_words = list(set(token_dict.values()))

        tokenizer = SimpleTokenizer(token_dict) # 建立分词器

        # keep_flags = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']

        rel2id = {rel: _id + 1 for _id, rel in enumerate(sorted(relationships))}
        rel2id['unk'] = 0

        if not incremental_train:
            with open(os.path.join(model_save_path, 'tokenizer.pkl'), "wb") as f:
                pickle.dump(tokenizer, f)

            with open(os.path.join(model_save_path, 'keep_words.pkl'), "wb") as f:
                pickle.dump(keep_words, f)

            with open(os.path.join(model_save_path, 'rel2id.pkl'), "wb") as f:
                pickle.dump(rel2id, f)

        self.tokenizer, self.keep_words, self.rel2id = tokenizer, keep_words, rel2id
        return tokenizer, keep_words, rel2id


    def batch_generator(self, data_file):
        X1, X2, Y = [], [], []
        class_weight_count = {}
        while True:
            temp_data_file = '{}.temp'.format(data_file)
            command = 'shuf {} -o {}'.format(data_file, temp_data_file)
            os.system(command)
            for (text, triple), (entity_lists, rel) in self.gernerator_data(temp_data_file):
                # print((text, triple), (entity_lists, rel))
                entity_lists = [entity for entity, entity_type in entity_lists]

                text = text[:self.input_length]
                y = self.rel2id.get(rel)
                y = to_categorical(y, num_classes=len(self.rel2id))
                for subject, predicate, object in triple:
                    if predicate != rel:
                        continue

                    # 有些类别的数据太多(最多类别记录有： 2476218， 最小类别记录仅454)，当数据量太多(超过最小标签类别数的100倍)是就按一定概率进行忽略；
                    class_weight_count.setdefault(rel, 0)
                    if random.random() < 0.8 and (class_weight_count.get(rel, 0) + 1) / (
                            min(class_weight_count.values()) + 1) > 10:
                        continue
                    else:
                        class_weight_count[rel] += 1

                    first_text = text.replace(subject, '')
                    x1, x2 = self.tokenizer.encode(first=first_text)

                    X1.append(x1)
                    X2.append(x2)
                    Y.append(y)
                    if len(X1) >= self.batch_size:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        Y = seq_padding(Y)
                        yield [X1, X2], Y
                        [X1, X2, Y] = [], [], []

                if random.random() < 1/(len(self.rel2id) * 2 + 1):
                    chat_rets = get_random_chat_question()
                    if chat_rets:
                        question = chat_rets[0]['question']
                        answer = chat_rets[0]['answer']
                        rel = 'unk'
                        y = self.rel2id.get(rel)
                        y = to_categorical(y, num_classes=len(self.rel2id))
                        for first_text in [question, answer]:
                            x1, x2 = self.tokenizer.encode(first=first_text)
                            X1.append(x1)
                            X2.append(x2)
                            Y.append(y)

            if X1:
                X1 = seq_padding(X1)
                X2 = seq_padding(X2)
                Y = seq_padding(Y)
                yield [X1, X2], Y
                [X1, X2, Y] = [], [], []

            os.system('rm {}'.format(temp_data_file))

    def build_model(self):
        model = load_pretrained_model(
            config_path,
            checkpoint_path,
            keep_words=self.keep_words,
            albert=True
        )
        class_num = len(self.rel2id)
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
        plot_model(model, 'triple_model.png')

        model.summary()
        return model

    def train(self, epochs=10):
        model = self.build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                                 'triple-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.hdf5'),
                                           save_best_only=True, save_weights_only=False)

        tb = TensorBoard(log_dir=log_dir,  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         batch_size=self.batch_size,  # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=False,  # 是否可视化梯度直方图
                         write_images=False,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

        hist = model.fit_generator(
            self.batch_generator(self.train_file),
            # batch_size=batch_size,
            epochs=epochs,
            # verbose=1,
            steps_per_epoch=500,
            # validation_split=0.1,
            validation_data= self.batch_generator(self.dev_file),
            validation_steps=50,
            shuffle=True,
            # class_weight={'ner_out': 'auto', 'rel_out': 'auto'},
            callbacks=[early_stopping, model_checkpoint, tb]
            )

        print(hist.history.items())

    def incremental_train(self, filepath='classify-02-0.62-0.831.hdf5', epochs=10, initial_epoch=0):
        """
        加载模型增量训练
        :param filepath:
        :return
        """
        print('开始增量训练...')
        model = self.build_model()
        model.load_weights(os.path.join(model_save_path, filepath), by_name=True,
                           skip_mismatch=True, reshape=True)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                                 'triple-{epoch:02d}-{acc:.2f}-{val_acc:.3f}.hdf5'),
                                           save_best_only=True, save_weights_only=False)

        tb = TensorBoard(log_dir=log_dir,  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         batch_size=self.batch_size,  # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=False,  # 是否可视化梯度直方图
                         write_images=False,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

        hist = model.fit_generator(
            self.batch_generator(self.train_file),
            # batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            # verbose=1,
            steps_per_epoch=500,
            # validation_split=0.1,
            validation_data= self.batch_generator(self.dev_file),
            validation_steps=50,
            shuffle=True,
            # class_weight={'ner_out': 'auto', 'rel_out': 'auto'},
            callbacks=[early_stopping, model_checkpoint, tb]
            )

        print(hist.history.items())

    def predict(self, text, filepath='classify-25-0.41-0.868.hdf5'):
        text = text[:maxlen]
        triples = search_three_tuple(text)
        print('es查询结果：{}'.format(triples))
        triples = [[subject, predicate, object] for subject, predicate, object in triples if subject in text]
        id2label = {v: k for k, v in self.rel2id.items()}

        model = self.build_model()
        model.load_weights(os.path.join(model_save_path, filepath), by_name=True,
                           skip_mismatch=True, reshape=True)

        X1, X2 = [], []
        x1, x2 = self.tokenizer.encode(first=text)
        X1.append(x1)
        X2.append(x2)

        for subject, predicate, object in triples:
            first_text = text.replace(subject, '')
            x1, x2 = self.tokenizer.encode(first=first_text)
            X1.append(x1)
            X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        rets = model.predict([X1, X2])
        answer = []
        ret0 = rets[0]
        sort_index = ret0.argsort()
        label_score = [[id2label.get(_index), float(ret0[_index])] for _index in sort_index[-3:][::-1]]
        print('`{}`的属性：{}'.format(text, label_score))
        for ret, (subject, predicate, object) in zip(rets[1: ], triples):
            # print(ret)
            # leabel = id2label.get(ret.argmax())
            top1_score = ret[ret.argmax()]
            # print('top1_score: {}'.format(top1_score))
            if top1_score < 0.25:
                continue
            sort_index = ret.argsort()
            label_score = [[id2label.get(_index), float(ret[_index])] for _index in sort_index[-3:][::-1]]
            print(subject, predicate, label_score)
            labels = [label for label, score in label_score]
            if predicate in labels:
                answer.append([subject, predicate, object])
        answer_str = '\n'.join('\t'.join(triple) for triple in answer)
        return answer_str

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        is_train = True
    else:
        is_train = False

    triple_model = TripleModel(train=is_train)

    if len(sys.argv) > 1 and sys.argv[1] == 'incremental_train':
        triple_model.incremental_train(filepath='triple-07-0.9729-0.9844.hdf5')
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        triple_model.train(epochs=50)
    else:
        ret = triple_model.predict(sys.argv[1], filepath='triple-07-0.9729-0.9844.hdf5')
        print(ret)

if __name__ == '__main__':
    main()


