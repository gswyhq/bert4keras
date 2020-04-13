#!/usr/bin/python3
# coding: utf-8

import os, sys
import pickle
import random
import unicodedata
from examples.ner_classify_albert_tiny import Data_set

# 由原始数据生成增强的数据；
model_save_path = './models_ner_classify_albert_tiny20191030_1809稳定版本'

TRAIN_DATA_PATH = "./data/ner_rel_train_BIOES.txt.temp"
DEV_DATA_PATH = "./data/ner_rel_dev_BIOES.txt.temp"
TEST_DATA_PATH = "./data/ner_rel_test_BIOES.txt"

input_length=200

def generator_augmentation_data(temp_data_file, augmentation_file):
    pre_data = Data_set()

    # with open(os.path.join(model_save_path, 'keep_words.pkl'), "rb") as f:
    #     keep_words = pickle.load(f)
    with open(os.path.join(model_save_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_save_path, 'flag2id.pkl'), "rb") as f:
        flag2id = pickle.load(f)
    with open(os.path.join(model_save_path, 'rel2id.pkl'), "rb") as f:
        rel2id = pickle.load(f)

    pre_data.tokenizer, pre_data.flag2id, pre_data.rel2id = tokenizer, flag2id, rel2id

    class_weight_count = {}
    _index_line = 0
    with open(augmentation_file, 'w', errors='ignore')as fw:
        with open(temp_data_file, "r") as f:
            line = f.readline()
            while line:
                _index_line += 1
                if _index_line % 10000 == 0:
                    print(_index_line)
                # print(line)
                line = unicodedata.normalize('NFKD', line).strip()
                if '/' in line:
                    word_flag, rel_tag = [[word.rsplit('/', maxsplit=1) for word in line.rsplit('\t', maxsplit=1)[0].split() if
                                           word[1] == '/'], line.rsplit('\t', maxsplit=1)[-1]]

                    # 80%概率忽略掉30个字符以上的，80%概率忽略掉没有实体词的语料
                    if (random.random() > 0.2 and len(word_flag) > 30) or (
                            random.random() > 0.2 and all(flag == 'O' for word, flag in word_flag[:20])):
                        line = f.readline()
                        continue

                    word_flag = word_flag[:input_length]

                    # 有些类别的数据太多(最多类别记录有： 2476218， 最小类别记录仅454)，当数据量太多(超过最小标签类别数的100倍)是就按一定概率进行忽略；
                    class_weight_count.setdefault(rel_tag, 0)
                    if random.random() < 0.8 and (class_weight_count.get(rel_tag, 0) + 1) / (
                            min(class_weight_count.values()) + 1) > 10:
                        line = f.readline()
                        continue
                    else:
                        class_weight_count[rel_tag] += 1

                    if random.random() > 0.2 and input_length - len(word_flag) > 20 and \
                            any(word for word, flag in word_flag if flag == 'O'):
                        word_flag = pre_data.data_augmentation(word_flag, rel_tag)

                    fw.write('{}\t{}\n'.format(' '.join(['/'.join(wf) for wf in word_flag]), rel_tag))
                    # print('{}\t{}\n'.format(' '.join(['/'.join(wf) for wf in word_flag]), rel_tag))
                    # sys.exit(0)
                line = f.readline()

def main():
    generator_augmentation_data(DEV_DATA_PATH, DEV_DATA_PATH+'.aug')
    generator_augmentation_data(TRAIN_DATA_PATH, TRAIN_DATA_PATH+'.aug')


if __name__ == '__main__':
    main()