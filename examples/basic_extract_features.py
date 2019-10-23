#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征

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
model = load_pretrained_model(config_path, checkpoint_path) # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

"""
输出：
[[[-0.63251007  0.2030236   0.07936534 ...  0.49122632 -0.20493352
    0.2575253 ]
  [-0.7588351   0.09651865  1.0718756  ... -0.6109694   0.04312154
    0.03881441]
  [ 0.5477043  -0.792117    0.44435206 ...  0.42449304  0.41105673
    0.08222899]
  [-0.2924238   0.6052722   0.49968526 ...  0.8604137  -0.6533166
    0.5369075 ]
  [-0.7473459   0.49431565  0.7185162  ...  0.3848612  -0.74090636
    0.39056838]
  [-0.8741375  -0.21650358  1.338839   ...  0.5816864  -0.4373226
    0.56181806]]]
"""
