#!/usr/bin/python3
# coding: utf-8

import os

# albert_config_large.json  albert_model.ckpt.data-00000-of-00001  albert_model.ckpt.index  albert_model.ckpt.meta  checkpoint  vocab.txt
albert_model_path = './albert_large_zh'
# albert_model_path = '/notebooks/albert_zh/albert_large_zh'
# https://storage.googleapis.com/albert_zh/albert_large_zh.zip

config_path = os.path.join(albert_model_path, 'albert_config_large.json')
checkpoint_path = os.path.join(albert_model_path, 'albert_model.ckpt')
dict_path = os.path.join(albert_model_path, 'vocab.txt')

# albert_model_path = './albert_tiny'
# config_path = os.path.join(albert_model_path, 'albert_config_tiny.json')

model_save_path = './models'
log_dir = './logs'

TERM_FREQUENCY_FILE = './data/微博词频统计.json'

def main():
    pass


if __name__ == '__main__':
    main()