#!/usr/bin/python3
# coding: utf-8

import os
import sys
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.layers import Embedding, Bidirectional, LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

BATCH_SIZE = 32
# MODEL_PATH = "./model/crf.h5"
model_save_path = './model'
log_dir = './logs'
LABELS_CATEGORY = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
TRAIN_DATA_PATH = "./data/example.train"
DEV_DATA_PATH = "./data/example.dev"
TEST_DATA_PATH = "./data/example.test"
VOCAB_PATH = "./model/vocab.pk"
EPOCHS = 2

# 训练数据来源：
# https://github.com/EricZhu-42/NJU_NLP_SummerCamp_2019/tree/8067020bc706297d73a844adba4558a65bc4ca9a/Week_2/7.8-7.14_NER/data

class Data_set:
    def __init__(self, labels):
        self.labels = labels

    def load_data(self, data_path):

        with open(data_path, "rb") as f:
            data = f.read().decode("utf-8")
        process_data = self.processing_data(data)
        return process_data

    def processing_data(self, data):
        data = data.split("\n\n")
        data = [token.split("\n") for token in data]
        data = [[j.split() for j in i] for i in data]
        data.pop()
        return data

    def save_vocab(self, save_path, process_data):
        all_char = [char[0] for sen in process_data for char in sen]
        chars = set(all_char)
        word2id = {char: id_ + 1 for id_, char in enumerate(chars)}
        word2id["unk"] = 0
        with open(save_path, "wb") as f:
            pickle.dump(word2id, f)
        return word2id

    def generate_data(self, vocab, process_data, maxlen):
        char_data_sen = [[token[0] for token in i] for i in process_data]
        label_sen = [[token[1] for token in i] for i in process_data]
        sen2id = [[vocab.get(char, 0) for char in sen] for sen in char_data_sen]
        label2id = {label: id_ for id_, label in enumerate(self.labels)}
        lab_sen2id = [[label2id.get(lab, 0) for lab in sen] for sen in label_sen]
        sen_pad = pad_sequences(sen2id, maxlen)
        lab_pad = pad_sequences(lab_sen2id, maxlen, value=-1)
        lab_pad = np.expand_dims(lab_pad, 2)
        return sen_pad, lab_pad


class Ner:
    def __init__(self, labels_category, vocab=None, Embedding_dim=200, model_path='', train=True):
        self.Embedding_dim = Embedding_dim
        self.labels_category = labels_category
        if train:
            self.vocab = vocab
            self.model = self.build_model()
        else:
            with open(VOCAB_PATH, "rb") as f:
                self.vocab = pickle.load(f)
            self.model = load_model(model_path,
                           custom_objects={'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy})

    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.vocab), self.Embedding_dim, mask_zero=True))  # Random embedding
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        crf = CRF(len(self.labels_category), sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
        return model

    # 训练后保存模型
    def train(self, data, label, validation_data, epochs=EPOCHS):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                                 'checkpoint-{epoch:02d}-{val_loss:.2f}-{val_crf_accuracy:.3f}.hdf5'),
                                           save_best_only=True, save_weights_only=False)

        tb = TensorBoard(log_dir=log_dir,  # log 目录
                         histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         batch_size=32,  # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=False,  # 是否可视化梯度直方图
                         write_images=False,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

        history = self.model.fit(data, label,
                       batch_size=BATCH_SIZE,
                       epochs=epochs,
                       validation_data = validation_data,
                       callbacks=[early_stopping, model_checkpoint, tb]
                       )

        print('history: {}'.format(history.history))
        # self.model.save(MODEL_PATH)

    def predict(self, data, maxlen):
        char2id = [self.vocab.get(i) for i in data]
        # pad_num = maxlen - len(char2id)
        input_data = pad_sequences([char2id], maxlen)
        result = self.model.predict(input_data)[0][-len(data):]
        result_label = [LABELS_CATEGORY[np.argmax(i)] for i in result]
        # ['B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        return result_label

    def evaluate(self, data, label, batch_size=None):
        [loss, acc] = self.model.evaluate(data, label, batch_size=batch_size)

        return loss, acc

def predict(data, maxlen=200):
    ner = Ner(LABELS_CATEGORY, train=False, model_path='./model/checkpoint-02-9.13-0.966.hdf5')

    result_label = ner.predict(data, maxlen)
    # print(result_label)
    return result_label

def train():
    """训练"""
    data = Data_set(LABELS_CATEGORY)
    train_data = data.load_data(TRAIN_DATA_PATH)
    train_data = train_data
    dev_data = data.load_data(DEV_DATA_PATH)
    vocab = data.save_vocab(VOCAB_PATH, train_data+dev_data)
    sentence, sen_tags = data.generate_data(vocab, train_data, 200)
    validation_data = data.generate_data(vocab, dev_data, 200)
    ner = Ner(LABELS_CATEGORY, vocab=vocab, train=True)
    print('validation_data: {}'.format(len(validation_data[0])))
    ner.train(sentence, sen_tags, validation_data, EPOCHS)

def main():
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = 'train'
    if text == 'train':
        print('开始训练模型。。。')
        train()
    else:
        ret = predict(text, maxlen=200)
        print("`{}`的命名实体识别的结果：{}".format(text, ret))

if __name__ == '__main__':
    main()
