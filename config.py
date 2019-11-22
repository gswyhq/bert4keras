#!/usr/bin/python3
# coding: utf-8

import os

ES_HOST = os.getenv('ES_HOST', '192.168.3.105')
ES_PORT = os.getenv('ES_PORT', '9200')
ES_USER = os.getenv('ES_USER', 'elastic')
ES_PASSWORD = os.getenv('ES_PASSWORD', 'changeme')
ES_TIMEOUT = (0.5, 10)

# albert_config_large.json  albert_model.ckpt.data-00000-of-00001  albert_model.ckpt.index  albert_model.ckpt.meta  checkpoint  vocab.txt
# albert_model_path = './albert_large_zh'
albert_model_path = './albert_tiny'
# albert_model_path = '/notebooks/albert_zh/albert_large_zh'
# https://storage.googleapis.com/albert_zh/albert_large_zh.zip

# config_path = os.path.join(albert_model_path, 'albert_config_large.json')
config_path = os.path.join(albert_model_path, 'albert_config_tiny.json')
checkpoint_path = os.path.join(albert_model_path, 'albert_model.ckpt')
dict_path = os.path.join(albert_model_path, 'vocab.txt')

WORD_EMBEDDING_FILE = './data/Tencent_char_Embedding.txt'

model_save_path = './models'
log_dir = './logs'

TERM_FREQUENCY_FILE = './data/微博词频统计.json'

def main():
    pass


if __name__ == '__main__':
    main()