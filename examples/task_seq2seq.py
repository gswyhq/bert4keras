#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933

from __future__ import print_function
import glob
import numpy as np
from tqdm import tqdm
import os, json, codecs
from collections import Counter
# import uniout
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


config_path = 'seq2seq_config.json'
min_count = 32
max_input_len = 256
max_output_len = 32
batch_size = 16
steps_per_epoch = 1000
epochs = 10000


def read_text():
    txts = glob.glob('../THUCNews/*/*.txt')
    np.random.shuffle(txts)
    for txt in txts:
        d = open(txt).read()
        d = d.decode('utf-8').replace(u'\u3000', ' ')
        d = d.split('\n')
        if len(d) > 1:
            title = d[0].strip()
            content = '\n'.join(d[1:]).strip()
            if len(title) <= max_output_len:
                yield content[:max_input_len], title


if os.path.exists(config_path):
    chars = json.load(open(config_path))
else:
    chars = {}
    for a in tqdm(read_text(), desc=u'构建字表中'):
        for b in a:
            for w in b: # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1
    chars = [(i, j) for i, j in chars.items() if j >= min_count]
    chars = sorted(chars, key=lambda c: - c[1])
    chars = [c[0] for c in chars]
    json.dump(
        chars,
        codecs.open(config_path, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


albert_model_path = '/home/gswyhq/github_projects/albert_zh/albert_large_zh'
# albert_model_path = '/notebooks/albert_zh/albert_large_zh'
# https://storage.googleapis.com/albert_zh/albert_large_zh.zip

config_path = os.path.join(albert_model_path, 'albert_config_large.json')
checkpoint_path = os.path.join(albert_model_path, 'albert_model.ckpt')
dict_path = os.path.join(albert_model_path, 'vocab.txt')

# config_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt'


_token_dict = load_vocab(dict_path) # 读取词典
token_dict, keep_words = {}, set()

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.add(_token_dict[c])

for c in chars:
    if c in _token_dict:
        token_dict[c] = len(token_dict)
        keep_words.add(_token_dict[c])

keep_words.add(max(keep_words)+1)
keep_words = list(keep_words)

tokenizer = SimpleTokenizer(token_dict) # 建立分词器


def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def data_generator():
    while True:
        X, S = [], []
        for a, b in read_text():
            x, s = tokenizer.encode(a, b)
            X.append(x)
            S.append(s)
            if len(X) == batch_size:
                X = padding(X)
                S = padding(S)
                yield [X, S], None
                X, S = [], []


model = load_pretrained_model(
    config_path,
    checkpoint_path,
    seq2seq=True,
    keep_words=keep_words
)

model.summary()

# 交叉熵作为loss，并mask掉输入部分的预测
y_in = model.input[0][:, 1:] # 目标tokens
y_mask = model.input[1][:, 1:]
y = model.output[:, :-1] # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    target_ids = [[] for _ in range(topk)] # 候选答案id
    target_scores = [0] * topk # 候选答案分数
    for i in range(max_output_len): # 强制要求输出不超过max_output_len字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model.predict(
            [_target_ids, _segment_ids]
        )[:, -1, 3:] # 直接忽略[PAD], [UNK], [CLS]
        _log_probas = np.log(_probas + 1e-6) # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:] # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:] # 从中选出新的topk
        for j, k in enumerate(_topk_arg):
            target_ids[j].append(_candidate_ids[k][-1])
            target_scores[j] = _candidate_scores[k]
        ends = [j for j, k in enumerate(target_ids) if k[-1] == 3]
        if len(ends) > 0:
            k = np.argmax([target_scores[j] for j in ends])
            return tokenizer.decode(target_ids[ends[k]])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时 就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', gen_sent(s))
    print()


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()


if __name__ == '__main__':

    evaluator = Evaluate()

    model.fit_generator(
        data_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
