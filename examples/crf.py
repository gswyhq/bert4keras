"""Train CRF and BiLSTM-CRF on CONLL2000 chunking data,
similar to https://arxiv.org/pdf/1508.01991v1.pdf.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import pickle
import numpy as np
from collections import Counter
import tensorflow.keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Bidirectional, LSTM, Lambda, Conv1D, Dropout, concatenate, Input
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

import tensorflow as tf
from keras.layers.normalization import BatchNormalization

sys.path.append('.')

import ner_data

EPOCHS = 10
EMBED_DIM = 200
BiRNN_UNITS = 200
MAXLEN = 128

# gswyhq@gswyhq-PC:~/github_projects/bert4keras/data$ head -n 50000 ner_rel_train_BIOES.txt.temp > ner_rel_train_BIOES_5w.txt
# gswyhq@gswyhq-PC:~/github_projects/bert4keras/data$ head -n 5000 ner_rel_dev_BIOES.txt.temp > ner_rel_dev_BIOES_5k.txt

# TRAIN_PATH = 'data/ner_rel_train_BIOES_5w.txt'
# TEST_PATH = 'data/ner_rel_dev_BIOES_5k.txt'

TRAIN_PATH = 'data/ner_rel_train_BIOES.txt.temp.aug2'
TEST_PATH = 'data/ner_rel_dev_BIOES.txt.temp.aug2'

MODEL_SAVE_PATH = './models_bilstm_crf'
VOCAB_FILE = os.path.join(MODEL_SAVE_PATH, 'crf_voc.pkl')

def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics,
    reports per classs recall, precision and F1 score'''
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    '召回率（Recall）',
                                                    '精确率（Precision）',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')


# ------
# Data
# -----

if len(sys.argv) > 1 and sys.argv[1].startswith('train'):
    train, test, voc = ner_data.load_data(train_path=TRAIN_PATH, test_path=TEST_PATH, maxlen=MAXLEN)
    (train_x, train_y) = train
    (test_x, test_y) = test
    (vocab, class_labels) = voc

    with open(VOCAB_FILE, 'wb')as f:
        pickle.dump(voc, f)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_PATH, 'ner-{epoch:02d}-{val_crf_viterbi_accuracy:.4f}.hdf5'),
                            save_best_only=True, save_weights_only=False)

# --------------
# 1. Regular CRF
# --------------
def regular_crf(train_x, train_y, test_x, test_y):
    print('==== training CRF ====')
    
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    # The default `crf_loss` for `learn_mode='join'` is negative log likelihood.
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])

    model.fit(train_x, train_y, epochs=EPOCHS,
              validation_data=[test_x, test_y],
              callbacks=[early_stopping, model_checkpoint])
    
    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]
    
    print('\n---- Result of CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)

# -------------
# 2. BiLSTM-CRF
# -------------

def bilstm_crf(train_x, train_y, test_x, test_y):
    print('==== training BiLSTM-CRF ====')
    
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()
    
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y],
              callbacks=[early_stopping, model_checkpoint])
    
    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]
    
    print('\n---- Result of BiLSTM-CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)

def result_to_json(string, tags):
    # print('string: {}, tags: {}'.format(string, tags))
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

def bilstm_crf_predict(data, vocab_file=VOCAB_FILE, model_file=''):
    """
    加载模型进行预测
    :param data: ['刘舒曼是哪个民族的', '姚雪华的出生地是哪里', '请问珠穆朗玛峰有多高','我想知道太平洋有多大']
    :return:
    """
    model = load_model(model_file,
                       custom_objects={'CRF': CRF,
                            'crf_loss': crf_loss,
                            'crf_viterbi_accuracy': crf_viterbi_accuracy
                                       }
                       )

    with open(vocab_file, 'rb')as f:
        voc = pickle.load(f)

    (vocab, class_labels) = voc

    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
    x = pad_sequences(x, MAXLEN)  # left padding
    rets = model.predict(x).argmax(-1)
    rets = [k[v>0] for k, v in zip(rets, x)]
    results = [result_to_json(string, [class_labels[int(i)] for i in tags_ids]) for string, tags_ids in zip(data, rets)]

    # 从测试集中读取数据进行评估
    with open(TEST_PATH)as f:
        acc_count = 0
        total_count = 0
        datas = [t.rsplit('\t', maxsplit=1)[0].split() for t in f.readlines()]
        for data_flag in [datas[i*32:(i+1) * 32] for i in range(0, int(len(datas)/32), 1)]:
            data = [''.join([t[0] for t in line if t[1] == '/']) for line in data_flag]

            x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
            x = pad_sequences(x, MAXLEN)  # left padding
            rets = model.predict(x).argmax(-1)
            rets = [k[v > 0] for k, v in zip(rets, x)]
            ret3 = [1 if all(k[0]==v[0] for k, v in zip([t[2:].upper() for t in str_tag if t[1] == '/'], [class_labels[int(i)] for i in tags_ids])) else 0 for str_tag, tags_ids in zip(data_flag, rets)]
            acc_count += sum(ret3)
            total_count += len(ret3)
            print('正确率：{}'.format(acc_count/(total_count+0.0001)))
            if total_count > 20000:
                sys.exit(0)
    return results

# -------------
# 3. IDCNN-CRF
# -------------

class MaskedConv1D(Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None and self.padding == 'valid':
            mask = mask[:, self.kernel_size[0] // 2 * self.dilation_rate[0] * 2:]
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)

def IDCNN(input, cnn_filters=128, cnn_kernel_size=3, cnn_blocks=4, **kwargs):
    def _dilation_conv1d(dilation_rate):
        return MaskedConv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, padding="same", dilation_rate=dilation_rate)

    def _idcnn_block():
        idcnn_1 = _dilation_conv1d(1)
        idcnn_2 = _dilation_conv1d(1)
        idcnn_3 = _dilation_conv1d(2)
        return [idcnn_1, idcnn_2, idcnn_3]

    input = BatchNormalization(name='normalization')(input)

    stack_idcnn_layers = []
    for layer_idx in range(cnn_blocks):
        idcnn_block = _idcnn_block()
        cnn = idcnn_block[0](input)
        cnn = Dropout(0.02)(cnn)
        cnn = idcnn_block[1](cnn)
        cnn = Dropout(0.02)(cnn)
        cnn = idcnn_block[2](cnn)
        cnn = Dropout(0.02)(cnn)
        stack_idcnn_layers.append(cnn)
    stack_idcnn = concatenate(stack_idcnn_layers, axis=-1)
    return stack_idcnn

def seq_padding(X, padding=0, max_len=100):
    if len(X.shape) == 2:
        return np.array([
            np.concatenate([[padding] * (max_len - len(x)), x]) if len(x) < max_len else x for x in X
        ])
    elif len(X.shape) == 3:
        return np.array([
            np.concatenate([[[padding]] * (max_len - len(x)), x]) if len(x) < max_len else x for x in X
        ])
    else:
        return X

def idcnn_crf(train_x, train_y, test_x, test_y):
    test_x = seq_padding(test_x, padding=0, max_len=train_x.shape[1])
    test_y = seq_padding(test_y, padding=-1, max_len=train_y.shape[1])
    
    print('==== training IDCNN-CRF ====')
    
    # build models
    input = Input(shape=(train_x.shape[-1],))
    emb = Embedding(len(vocab), EMBED_DIM, mask_zero=True)(input)
    idcnn = IDCNN(emb)
    crf_out = CRF(len(class_labels), sparse_target=True)(idcnn)
    model = Model(input, crf_out)
    model.summary()

    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y],
              callbacks=[early_stopping, model_checkpoint])
    
    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]
    
    print('\n---- Result of IDCNN-CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)

def idcnn_crf_predict(data, vocab_file=VOCAB_FILE, model_file=''):
    """
    加载模型进行预测
    :param data: ['刘舒曼是哪个民族的', '姚雪华的出生地是哪里', '请问珠穆朗玛峰有多高','我想知道太平洋有多大']
    :return:
    """
    model = load_model(model_file,
                       custom_objects={'CRF': CRF,
                            'crf_loss': crf_loss,
                            'crf_viterbi_accuracy': crf_viterbi_accuracy,
                            'MaskedConv1D': MaskedConv1D,
                                       }
                       )

    with open(vocab_file, 'rb')as f:
        voc = pickle.load(f)

    (vocab, class_labels) = voc

    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
    x = pad_sequences(x, MAXLEN)  # left padding
    rets = model.predict(x).argmax(-1)
    rets = [k[v>0] for k, v in zip(rets, x)]
    results = [result_to_json(string, [class_labels[int(i)] for i in tags_ids]) for string, tags_ids in zip(data, rets)]
    return results

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train_crf':
        print('开始训练crf模型...')
        regular_crf(train_x, train_y, test_x, test_y)

    elif len(sys.argv) > 1 and sys.argv[1] == 'train_bilstm_crf':
        print('开始训练bilstm-crf模型...')
        bilstm_crf(train_x, train_y, test_x, test_y)

    elif len(sys.argv) > 1 and sys.argv[1] == 'train_idcnn_crf':
        print('开始训练idcnn-crf模型...')
        idcnn_crf(train_x, train_y, test_x, test_y)

    # 预测
    elif len(sys.argv) > 1 and sys.argv[1] == 'predict_bilstm_crf':
        data = ['刘舒曼是哪个民族的', '姚雪华的出生地是哪里', '请问珠穆朗玛峰有多高','我想知道太平洋有多大']
        ret = bilstm_crf_predict(data, vocab_file='models_bilstm_crf/crf_voc.pkl', model_file='models_bilstm_crf/ner-01-0.9231.hdf5')
        print([(k, v) for k, v in zip(data, ret)])

    elif len(sys.argv) > 1 and sys.argv[1] == 'predict_idcnn_crf':
        data = ['刘舒曼是哪个民族的', '姚雪华的出生地是哪里', '请问珠穆朗玛峰有多高','我想知道太平洋有多大']
        ret = idcnn_crf_predict(data, vocab_file='models_idcnn_crf/crf_voc.pkl', model_file='models_idcnn_crf/ner-07-0.9269.hdf5')
        print([(k, v) for k, v in zip(data, ret)])
    else:
        print('参数不对')
    
if __name__ == '__main__':
    main()

# gswyhq@gswyhq-PC:~/github_projects/bert4keras$ nohup python3 examples/crf.py > train_crf.log &
# python3 examples/crf.py train_idcnn_crf > train_idcnn_crf.log
