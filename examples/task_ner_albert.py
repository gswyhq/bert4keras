#!/usr/bin/python3
# coding: utf-8

import os
import sys
import pickle
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.layers import Embedding, Bidirectional, LSTM, Lambda, Dense
from keras.utils import plot_model
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

import sys
sys.path.append('.')

from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.utils import SimpleTokenizer, load_vocab
# from bert4keras.train import PiecewiseLinearLearningRate

from config import model_save_path, config_path, checkpoint_path, log_dir, dict_path

set_gelu('tanh') # 切换gelu版本

# 禁用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


BATCH_SIZE = 32

model_save_path = './models_ner'
# MODEL_PATH = "./model/crf.h5"
# model_save_path = './model'
# log_dir = './logs'
# LABELS_CATEGORY = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

TRAIN_DATA_PATH = "./data/example.train"
DEV_DATA_PATH = "./data/example.dev"
TEST_DATA_PATH = "./data/example.test"

EPOCHS = 2

# 训练数据来源：
# https://github.com/EricZhu-42/NJU_NLP_SummerCamp_2019/tree/8067020bc706297d73a844adba4558a65bc4ca9a/Week_2/7.8-7.14_NER/data

class Data_set:
    def __init__(self):
        pass

    def load_data(self, data_path):

        with open(data_path, "rb") as f:
            data = f.read().decode("utf-8")
        process_data = self.processing_data(data)
        return process_data

    def processing_data(self, data):
        if '\n\n' in data:
            data = data.split("\n\n")
            data = [token.split("\n") for token in data]
            data = [[j.split() for j in i] for i in data]
            data.pop()
        else:
            data = [[word.rsplit('/', maxsplit=1) for word in text.split() if word[1] == '/'] for text in data.split('\n') if '/' in text]
            data.pop()
        random.shuffle(data)
        return data

    def save_vocab(self, model_save_path, process_data):
        chars = set()
        labels = set()
        for char_labels in process_data:
            for char, label in char_labels:
                chars.add(char)
                labels.add(label)

        _token_dict = load_vocab(dict_path)  # 读取词典
        token_dict, keep_words = {}, set()

        for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
            token_dict[c] = len(token_dict)
            keep_words.add(_token_dict[c])

        for c in chars:
            if c in _token_dict:
                token_dict[c] = len(token_dict)
                keep_words.add(_token_dict[c])

        keep_words.add(max(keep_words) + 1)
        keep_words = list(keep_words)
        tokenizer = SimpleTokenizer(token_dict)  # 建立分词器

        with open(os.path.join(model_save_path, 'tokenizer.pkl'), "wb") as f:
            pickle.dump(tokenizer, f)

        with open(os.path.join(model_save_path, 'keep_words.pkl'), "wb") as f:
            pickle.dump(keep_words, f)

        # print('labels={}'.format(labels))
        # sorted: 保证 非实体词 O 的id为0
        self.label2id = {label: id_ for id_, label in enumerate(sorted(labels, key=lambda x: 0 if x == 'O' else 1))}
        print('label2id: {}'.format(self.label2id))
        with open(os.path.join(model_save_path, 'label2id.pkl'), "wb") as f:
            pickle.dump(self.label2id, f)

        self.keep_words = keep_words
        self.tokenizer = tokenizer

    # def generate_data(self, vocab, process_data, maxlen):
    #     char_data_sen = [[token[0] for token in i] for i in process_data]
    #     label_sen = [[token[1] for token in i] for i in process_data]
    #     sen2id = [[vocab.get(char, 0) for char in sen] for sen in char_data_sen]
    #     label2id = {label: id_ for id_, label in enumerate(self.labels)}
    #     lab_sen2id = [[label2id.get(lab, 0) for lab in sen] for sen in label_sen]
    #     sen_pad = pad_sequences(sen2id, maxlen)
    #     lab_pad = pad_sequences(lab_sen2id, maxlen, value=-1)
    #     lab_pad = np.expand_dims(lab_pad, 2)
    #     return sen_pad, lab_pad

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=32, maxlen=None, tokenizer=None, label2id=None):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.label2id = label2id
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = np.array([i for i in range(len(self.data))])
            # np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            # questions = []
            for i in idxs:
                d = self.data[i]
                text = ''.join([w for w, t in d[:self.maxlen]])
                x1, x2 = self.tokenizer.encode(first=text)
                # y = [to_categorical(self.label2id.get(label), num_classes=len(self.label2id)) for w, label in d[:self.maxlen]]
                y = [self.label2id.get('O')] + [self.label2id.get(label) for w, label in d[:self.maxlen]] + [self.label2id.get('O')]
                # y = np.array([to_categorical(self.label2id.get(label), num_classes=len(self.label2id)) for w, label in d[:self.maxlen]])
                # y = np.expand_dims(y, 2)
                # x1 = pad_sequences(x1, self.maxlen)
                # x2 = pad_sequences(x2, self.maxlen)
                # y = np.expand_dims(y, 2)

                # print('x1: {}, x2: {}, y: {}'.format(x1, x2, y))
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                # questions.append(d)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    # print('x.shape: {}, y.shape: {}'.format(X1[0].shape, Y[0].shape))
                    Y = np.expand_dims(Y, 2)
                    # if X1.shape[-1] != self.maxlen + 2 or X2.shape[-1] != self.maxlen + 2:
                    #     print("d: {}".format(questions))
                    #
                    # print('X1: {}, X2: {}, Y: {}'.format(X1.shape, X2.shape, Y.shape))
                    # print('X1: {}, X2: {}, Y: {}'.format(X1[0].shape, X2[0].shape, Y[0].shape))
                    # print('退出')
                    # exit(0)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []
                    # questions = []

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

class Ner:
    def __init__(self, model_path='', train=True, tokenizer=None, keep_words=None, label2id=None):

        if train:
            self.tokenizer=tokenizer
            self.keep_words=keep_words
            self.label2id=label2id

        else:
            with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
                self.tokenizer = pickle.load(f)

            with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
                self.keep_words = pickle.load(f)

            with open(os.path.join(model_save_path, 'label2id.pkl'), "rb") as f:
                self.label2id = pickle.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
            self.model = self.build_model()

            self.model.load_weights(model_path, by_name=True,
                               skip_mismatch=True, reshape=True)

    def build_model(self):

        model = load_pretrained_model(
            config_path,
            checkpoint_path,
            keep_words=self.keep_words,
            albert=True
        )

        # output = Lambda(lambda x: x[:, 0])(model.output)
        output = CRF(len(self.label2id), sparse_target=True)(model.output)
        model = Model(model.input, output)

        model.compile(
            loss=crf_loss,
            optimizer=Adam(1e-5),  # 用足够小的学习率
            # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
            metrics=[crf_accuracy]
        )

        # 保存模型图
        plot_model(model, 'ner-albert.png')

        model.summary()
        # print('model.input_shape: {}, model.output_shape: {}'.format(model.input_shape, model.output_shape))

        return model

    # 训练后保存模型
    def train(self, train_data, valid_data, maxlen=None, epochs=EPOCHS):

        self.model = self.build_model()
        train_D = data_generator(train_data, maxlen=maxlen, tokenizer=self.tokenizer, label2id=self.label2id)
        valid_D = data_generator(valid_data, maxlen=maxlen, tokenizer=self.tokenizer, label2id=self.label2id)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                                 'checkpoint-{epoch:02d}-{val_loss:.2f}-{val_crf_accuracy:.3f}.hdf5'),
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

        # history = self.model.fit(data, label,
        #                batch_size=BATCH_SIZE,
        #                epochs=epochs,
        #                validation_data = validation_data,
        #                callbacks=[early_stopping, model_checkpoint, tb]
        #                )

        history = self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=epochs,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            shuffle=True,
            callbacks=[early_stopping, model_checkpoint, tb]
        )
        print('history: {}'.format(history.history))
        # self.model.save(MODEL_PATH)

    def predict(self, data, maxlen):
        X1 = []
        X2 = []
        for text in data:
            text = text[:maxlen]
            x1, x2 = self.tokenizer.encode(first=text)
            X1.append(x1)
            X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        rets = self.model.predict([X1, X2])
        result_label = [[self.id2label.get(r.argmax()) for r in ret[1:-1]] for ret in rets]
        # print(result_label)
        result = [result_to_json(text[:maxlen], tags) for text, tags in zip(data, result_label)]
        return result

    def evaluate(self, data, label, batch_size=None):
        [loss, acc] = self.model.evaluate(data, label, batch_size=batch_size)

        return loss, acc

def predict(data, maxlen=200):
    ner = Ner(train=False, model_path=os.path.join(model_save_path, 'checkpoint-02-0.14-0.960.hdf5'))

    result_label = ner.predict(data, maxlen)
    # print(result_label)
    return result_label

def train():
    """训练"""
    data = Data_set()
    train_data = data.load_data(TRAIN_DATA_PATH)
    train_data = train_data
    dev_data = data.load_data(DEV_DATA_PATH)
    data.save_vocab(model_save_path, train_data+dev_data)
    label2id = data.label2id
    keep_words = data.keep_words
    tokenizer = data.tokenizer
    # sentence, sen_tags = data.generate_data(vocab, train_data, 200)
    # validation_data = data.generate_data(vocab, dev_data, 200)
    ner = Ner(train=True, tokenizer=tokenizer, keep_words=keep_words, label2id=label2id)

    ner.train(train_data, dev_data, maxlen=50, epochs=EPOCHS)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print('开始训练模型...')
        train()
    else:
        text = sys.argv[1] if len(sys.argv) > 1 else '北京与上海的距离是是多少？'
        rets = predict([text], maxlen=50)
        print("`{}`的命名实体识别的结果：{}".format(text, rets[0]))

if __name__ == '__main__':
    main()
