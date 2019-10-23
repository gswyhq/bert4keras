#! -*- coding: utf-8 -*-
# 测试代码可用性: MLM

import os
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
import numpy as np

albert_model_path = '/home/gswyhq/github_projects/albert_zh/albert_large_zh'
# albert_model_path = '/notebooks/albert_zh/albert_large_zh'
# https://storage.googleapis.com/albert_zh/albert_large_zh.zip

config_path = os.path.join(albert_model_path, 'albert_config_large.json')
checkpoint_path = os.path.join(albert_model_path, 'albert_model.ckpt')
dict_path = os.path.join(albert_model_path, 'vocab.txt')

token_dict = load_vocab(dict_path) # 读取词典
tokenizer = SimpleTokenizer(token_dict) # 建立分词器
model = load_pretrained_model(config_path, checkpoint_path, with_mlm=True) # 建立模型，加载权重


# token_ids, segment_ids = tokenizer.encode(u'科学技术是第一生产力')
token_ids, segment_ids = tokenizer.encode(u'中国的首都是北京')

print('token_ids: {}, segment_ids: {}'.format(token_ids, segment_ids))

# mask掉“技术”
# token_ids[3] = token_ids[4] = token_dict['[MASK]']
token_ids[4] = token_ids[5] = token_dict['[MASK]']

# 用mlm模型预测被mask掉的部分
probas = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
# print(tokenizer.decode(probas[3:5].argmax(axis=1))) # 结果正是“技术”
print(tokenizer.decode(probas.argmax(axis=1)))
