#!/usr/bin/python3
# coding: utf-8

import os
import sys
import pickle
import json
import math
import itertools
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential

from keras.backend.tensorflow_backend import set_session
# from keras_contrib.layers import CRF
# from keras_contrib.metrics import crf_accuracy
# from keras_contrib.losses import crf_loss
from keras.layers import Embedding, Bidirectional, LSTM, Layer, Lambda, Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import keras.backend as K
from keras.utils.np_utils import to_categorical
sys.path.append('.')
from config import model_save_path, log_dir

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" # "0,1"

tf_config = tf.ConfigProto()
tf_config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

BATCH_SIZE = 32

TRAIN_DATA_PATH = "./data/train3.txt"
DEV_DATA_PATH = "./data/dev3.txt"
# TEST_DATA_PATH = "./data/test4.txt"

EPOCHS = 120

# 训练数据来源：
# https://github.com/EricZhu-42/NJU_NLP_SummerCamp_2019/tree/8067020bc706297d73a844adba4558a65bc4ca9a/Week_2/7.8-7.14_NER/data

class Data_set:
    def __init__(self):
        self.word2id = {}
        self.rel2id = {}
        self.train_count = 0
        self.dev_count = 0

    def generator_load_data(self, data_path, data_type='train'):
        # print('读取文件：{}'.format(data_path))
        with open(data_path, "r") as f:
            text = f.readline()
            while text:
                if '\t' in text:
                    data = text.strip().rsplit('\t', maxsplit=1)
                    if len(data) != 2:
                        text = f.readline()
                        continue
                    if data_type == 'train':
                        self.train_count += 1
                    elif data_type == 'dev':
                        self.dev_count += 1
                    yield data
                text = f.readline()

    def load_vocab(self, save_path):
        with open(os.path.join(save_path, 'word2id.pkl'), "rb") as f:
            word2id = pickle.load(f)
        with open(os.path.join(save_path, 'rel2id.pkl'), "rb") as f:
            rel2id = pickle.load(f)
        self.word2id, self.rel2id = word2id, rel2id

        return word2id, rel2id

    def save_vocab(self, save_path, process_data):
        relationships = set()
        chars = set()
        for text, label in process_data:
            chars.update(set(text))
            relationships.add(label)
        word2id = {char: id_ + 1 for id_, char in enumerate(chars)}
        word2id["unk"] = 0
        rel2id = {rel: _id for _id, rel in enumerate(relationships)}
        with open(os.path.join(save_path, 'word2id.pkl'), "wb") as f:
            pickle.dump(word2id, f)
        with open(os.path.join(save_path, 'rel2id.pkl'), "wb") as f:
            pickle.dump(rel2id, f)
        self.word2id, self.rel2id = word2id, rel2id
        return word2id, rel2id

    # def generate_data(self, process_data, maxlen):
    #     char_data_sen = [[token[0] for token in i] for i, j in process_data]
    #     sen2id = [[self.word2id.get(char, 0) for char in sen] for sen in char_data_sen]
    #     relel_sen = [to_categorical(self.rel2id.get(relel, 0), num_classes=len(self.rel2id)) for i, relel in process_data]
    #
    #
    #     sen_pad = pad_sequences(sen2id, maxlen)
    #     ner_pad = pad_sequences(ner_sen2id, maxlen, value=-1)
    #     ner_pad = np.expand_dims(ner_pad, 2)
    #     rel_pad = np.array(relel_sen)
    #     # shape: (200, 200), (200, 200, 1), (200, 42)
    #     print("head1: {}, {}, {}".format(sen_pad[0], ner_pad[0], rel_pad[0]))
    #     return sen_pad, ner_pad, rel_pad

    def batch_generator(self, data_file, batch_size=32, maxlen=200):
        batch_text = []
        batch_rel_tag = []
        while True:
            with open(data_file, "r") as f:
                text = f.readline()
                while text:
                    if '\t' in text:
                        data = text.strip().rsplit('\t', maxsplit=1)
                        if len(data) != 2:
                            text = f.readline()
                            continue
                        text, label = data
                        batch_text.append([self.word2id.get(word, 0)  for word in text])
                        batch_rel_tag.append(to_categorical(self.rel2id.get(label, 0), num_classes=len(self.rel2id)))
                        if len(batch_rel_tag) >= batch_size:
                            batch_sentence = pad_sequences(batch_text, maxlen)
                            batch_rel_tags = np.array(batch_rel_tag)
                            yield batch_sentence, batch_rel_tags
                            batch_text = []
                            batch_rel_tag = []
                    text = f.readline()
                if batch_rel_tag:
                    yield batch_sentence, batch_rel_tags

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

pre_data = Data_set()

class Classify():
    def __init__(self, word2id, rel2id, class_num=-1, input_dim=100, Embedding_dim=200, model_path='', train=True):
        self.Embedding_dim = Embedding_dim
        self.word2id = word2id
        self.rel2id = rel2id
        if train:
            self.model = self.build_model(input_dim, class_num)
        else:
            self.model = load_model(model_path, custom_objects={'Attention': Attention(128)})

    def build_model(self, input_dim, class_num):

        model = Sequential()
        model.add(Embedding(input_dim, self.Embedding_dim, mask_zero=True))  # Random embedding
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Lambda(lambda x: x, output_shape=lambda s: s))
        model.add(Attention(128))
        model.add(Lambda(lambda x: x[:, 0]))
        model.add(Dense(class_num, activation='softmax'))
        model.summary()
        model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # 训练后保存模型
    def train(self, input_length=200, epochs=EPOCHS, batch_size=BATCH_SIZE):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                                 'classify-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5'),
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

        # steps_per_epoch = math.ceil(pre_data.train_count / batch_size)
        # validation_steps = math.ceil(pre_data.dev_count / batch_size)

        hist = self.model.fit_generator(
            pre_data.batch_generator(TRAIN_DATA_PATH, batch_size=batch_size, maxlen=input_length),
            epochs=epochs,
            steps_per_epoch=10000,
            validation_data=pre_data.batch_generator(DEV_DATA_PATH, batch_size=batch_size, maxlen=input_length),
            validation_steps=1000,
            shuffle=True,
            class_weight='auto',
            callbacks=[early_stopping, model_checkpoint, tb]
            )
        # score = model.evaluate(x={'input': sentence},
        #                        y={'ner_out': flag_tags, 'rel_out': rel_tags},
        #                        batch_size=10, verbose=1)

        # print(score)
        print(hist.history.items())

    def predict(self, data, maxlen):
        sequences = []
        for text in data:
            sequences.append([self.word2id.get(word, 0) for word in text[:maxlen]])

        input_data = pad_sequences(sequences, maxlen)
        results = self.model.predict(input_data)
        result_label = []
        for text, rel_label in zip(data, results):
            sort_index = rel_label.argsort()
            result_label.append([{"intent": self.rel2id.get(_index), "score": float(sort_index[_index]), "question": text} for _index in sort_index[-3:][::-1]])

        return result_label

    def evaluate(self, data, label, batch_size=None):
        [loss, acc] = self.model.evaluate(data, label, batch_size=batch_size)

        return loss, acc

def predict(data, maxlen=200):
    word2id, rel2id = pre_data.load_vocab(model_save_path)
    classify = Classify(word2id, rel2id, class_num=len(rel2id), input_dim=len(word2id), train=False, model_path='./model/checkpoint-02-9.13-0.966.hdf5')

    result_label = classify.predict(data, maxlen)
    # print(result_label)
    return result_label


def train():
    """训练"""

    word2id, rel2id = pre_data.save_vocab(model_save_path, itertools.chain(pre_data.generator_load_data(TRAIN_DATA_PATH, data_type='train'),
                                                                   pre_data.generator_load_data(DEV_DATA_PATH, data_type='dev')))

    classify = Classify(word2id=word2id, rel2id=rel2id, class_num=len(rel2id), input_dim=len(word2id), train=True)
    print('训练集问题数：{}；验证集问题数：{}'.format(pre_data.train_count, pre_data.dev_count))
    classify.train(input_length=200, epochs=EPOCHS, batch_size=BATCH_SIZE)

def main():
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = 'train'
    if text == 'train':
        print('开始训练模型。。。')
        train()
    else:
        ret = predict([text], maxlen=200)
        print("`{}`的命名实体识别的结果：{}".format(text, ret))

if __name__ == '__main__':
    main()
