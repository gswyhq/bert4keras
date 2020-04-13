#!/usr/bin/python3
# coding: utf-8

# 输入问题及关联三元组，输出最佳三元组位置
import json
import os
import sys
import unicodedata
import itertools
import pickle
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Lambda
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from gensim.models import KeyedVectors
from keras.utils import plot_model
# pip3 install pydot -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com
# sudo apt-get install graphviz

sys.path.append('.')
from examples.ner_classify import result_to_json

WORD_EMBEDDING_FILE = './data/Tencent_char_Embedding.txt'

model_save_path = './models'
log_dir = './logs'
TRAIN_FILE = './data/ner_rel_train_BIOES.txt.temp.aug.triple2'
DEV_FILE = './data/ner_rel_dev_BIOES.txt.temp.aug.triple2'

class TripleModel():
    def __init__(self, train=False, train_file=TRAIN_FILE, dev_file=DEV_FILE):
        self.train_file = train_file
        self.dev_file = dev_file
        if train:
            self.build_word_embedding_weights()
        else:
            with open(os.path.join(model_save_path, 'word2id.pkl'), "rb") as f:
                self.word2id = pickle.load(f)
            with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
                self.rel2id = pickle.load(f)
            with open(os.path.join(model_save_path, 'word_embedding_weights.pkl'), "rb") as f:
                self.word_embedding_weights = pickle.load(f)

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

    def save_vocab(self, save_path, input_data):
        relationships = set()
        chars = set()
        for (text, triple), (entity_lists, rel) in input_data:
            chars.update(set(text))
            relationships.add(rel)
            relationships.update(set(p for s, p, o in triple))

        word2id = {char: id_ + 1 for id_, char in enumerate(chars)}
        word2id["unk"] = 0
        rel2id = {rel: _id + 1 for _id, rel in enumerate(relationships)}
        rel2id['unk'] = 0
        with open(os.path.join(save_path, 'word2id.pkl'), "wb") as f:
            pickle.dump(word2id, f)

        with open(os.path.join(save_path, 'rel2id.pkl'), "wb") as f:
            pickle.dump(rel2id, f)
        self.word2id, self.rel2id = word2id, rel2id
        return word2id, rel2id

    def build_word_embedding_weights(self):

        wv_from_text = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_FILE, binary=False)
        word2id, rel2id = self.save_vocab(model_save_path,
                                                       itertools.chain(self.gernerator_data(TRAIN_FILE),
                                                                       self.gernerator_data(DEV_FILE)))

        embedding_size = wv_from_text.vectors.shape[-1]
        word_embedding_weights = np.zeros((len(word2id), embedding_size))
        for word, i in word2id.items():
            if word in wv_from_text.index2word:
                word_embedding_weights[i] = wv_from_text.get_vector(word)

        with open(os.path.join(model_save_path, 'word_embedding_weights.pkl'), "wb") as f:
            pickle.dump(word_embedding_weights, f)

        self.word_embedding_weights = word_embedding_weights

    def batch_generator(self, data_file, batch_size=32, input_length=100, triple_size=20):
        batch_text = []
        triple_input = []
        main_output = []
        class_weight_count = {}
        # self.id2flag = {v: k for k, v in self.flag2id.items()}
        while True:
            temp_data_file = '{}.temp'.format(data_file)
            command = 'shuf {} -o {}'.format(data_file, temp_data_file)
            os.system(command)
            for (text, triple), (entity_lists, rel) in self.gernerator_data(temp_data_file):
                # print((text, triple), (entity_lists, rel))
                entity_lists = [entity for entity, entity_type in entity_lists]
                batch_text.append([self.word2id.get(word, 0)  for word in text[:input_length]])

                triple_ids = []
                output_id = 0
                for triple_index, (subject, predicate, object) in enumerate(triple):
                    if subject in entity_lists and rel == predicate:
                        output_id = triple_index
                    if subject in text:
                        entity_flag = 1
                    else:
                        entity_flag = 2
                    triple_id = to_categorical(self.rel2id.get(predicate, 0)*entity_flag, num_classes=len(self.rel2id) * 2)
                    triple_ids.append(triple_id)
                triple_ids = pad_sequences([triple_ids], triple_size, padding='post')
                # triple_ids = triple_ids.reshape(triple_ids.size, )
                triple_ids = triple_ids.reshape(triple_size, -1)
                # print("triple_ids.shape: {}".format(triple_ids.shape))
                triple_input.append(triple_ids)

                main_output.append(to_categorical(output_id, num_classes=triple_size))

                if len(batch_text) >= batch_size:
                    batch_text = pad_sequences(batch_text, input_length)
                    triple_input = np.array(triple_input)
                    main_output = np.array(main_output)
                    # print('batch_text.shape: {}, triple_input.shape: {}, main_output.shape: {}'.format(batch_text.shape, triple_input.shape, main_output.shape))
                    # print('batch_text: {}, triple_input: {}, main_output: {}'.format(batch_text, triple_input, main_output))
                    yield ({'text_input': batch_text, 'triple_input': triple_input}, {'main_output': main_output})
                    batch_text = []
                    triple_input = []
                    main_output = []

            if batch_text:
                yield ({'text_input': batch_text, 'triple_input': triple_input}, {'main_output': main_output})
                batch_text = []
                triple_input = []
                main_output = []

            os.system('rm {}'.format(temp_data_file))

    def build_model(self, maxlen=100, triple_size=20, train_word_embeddings=False):
        # construct model
        text_input = Input((maxlen,), dtype='int32', name='text_input')

        # x = Embedding(output_dim=512, input_dim=10000, input_length=100)(text_input)
        x = Embedding(input_dim=self.word_embedding_weights.shape[0],
                      output_dim=self.word_embedding_weights.shape[1],
                      weights=[self.word_embedding_weights],
                      trainable=train_word_embeddings,
                      mask_zero=True,
                      name='WordEmbedding')(text_input)

        lstm_out = Bidirectional(LSTM(100, return_sequences=True))(x)

        # bilstm_drop = Dropout(0.1)(lstm_out)
        # dense = TimeDistributed(Dense(word_embedding_weights.shape[1]))(bilstm_drop)
        # print('lstm_out.shape: {}'.format(lstm_out.shape))

        lstm_out = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_out)
        # lstm_out = Reshape((-1, ))(lstm_out)
        lstm_out = Dense(200, activation='relu')(lstm_out)

        # lstm_out = LSTM(32)(x)
        # aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

        triple_input = Input((triple_size, len(self.rel2id) * 2), name='triple_input')
        # triple_out = Reshape((-1,))(triple_input)
        triple_out = Dense(200, activation='relu')(triple_input)
        # rel_input = Input((triple_size, ), name='rel_input')
        # print('lstm_out: {}, triple_input: {}'.format(lstm_out.shape, triple_out.shape))
        x = keras.layers.concatenate([lstm_out, triple_out], axis=1)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Reshape((-1,))(x)
        main_output = Dense(triple_size, activation='softmax', name='main_output')(x)

        model = Model(inputs=[text_input, triple_input], outputs=[main_output])
        # model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
        model.compile(optimizer=Adam(lr=1e-4),
                      metrics={'main_output': 'accuracy'},
                      loss={'main_output': 'categorical_crossentropy'}
                      )
        print(model.summary())
        # 保存模型图
        plot_model(model, 'triple_model.png')

        return model

    def train(self, batch_size=32, input_length=100, triple_size=20, epochs=10):
        model = self.build_model(maxlen=input_length, triple_size=triple_size, train_word_embeddings=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                                 'triple-{epoch:02d}-{ner_out_crf_accuracy:.4f}-{val_rel_out_accuracy:.4f}.hdf5'),
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

        hist = model.fit_generator(
            self.batch_generator(self.train_file, batch_size=batch_size, input_length=input_length, triple_size=triple_size),
            # batch_size=batch_size,
            epochs=epochs,
            # verbose=1,
            steps_per_epoch=500,
            # validation_split=0.1,
            validation_data= self.batch_generator(self.dev_file, batch_size=batch_size, input_length=input_length, triple_size=triple_size),
            validation_steps=50,
            shuffle=True,
            # class_weight={'ner_out': 'auto', 'rel_out': 'auto'},
            callbacks=[early_stopping, model_checkpoint, tb]
            )

        print(hist.history.items())

# def multi_input_output_model():
#     # construct model
#     main_input = Input((100,), dtype='int32', name='main_input')
#
#     x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
#     lstm_out = LSTM(32)(x)
#     aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#
#     aux_input = Input((5,), name='aux_input')
#     x = keras.layers.concatenate([lstm_out, aux_input])
#     x = Dense(64, activation='relu')(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dense(64, activation='relu')(x)
#     main_output = Dense(1, activation='sigmoid', name='main_output')(x)
#
#     model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
#     # model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
#     model.compile(optimizer='rmsprop',
#                   loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
#                   loss_weights={'main_output': 1., 'aux_output': 0.3})
#
#     # 保存模型图
#     plot_model(model, 'Multi_input_output_model.png')
#
#     return model

def main():
    triple_model = TripleModel(train=True, train_file=TRAIN_FILE, dev_file=DEV_FILE)

    triple_model.train(batch_size=32, input_length=100, triple_size=20, epochs=10)
    # for i in triple_model.batch_generator(triple_model.train_file, batch_size=1, input_length=100, triple_size=20):
    #     break


if __name__ == '__main__':
    main()