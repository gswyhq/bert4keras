#!/usr/bin/python3
# coding: utf-8

import os
import sys
import unicodedata

def generator_load_data(data_path):
    # print('读取文件：{}'.format(data_path))
    with open(data_path, "r") as f:
        text = f.readline()
        while text:
            text = unicodedata.normalize('NFKD', text).strip()
            if '/' in text:
                text = text.strip()
                data = [[word.rsplit('/', maxsplit=1) for word in text.rsplit('\t', maxsplit=1)[0].split() if
                         word[1] == '/'], text.rsplit('\t', maxsplit=1)[-1]]

                yield data
            text = f.readline()

def bio2bioes(word_flag):
    """
    BIO标注格式转换为BIOES格式
    :param word_flag: [['谢', 'B-Shiyi'], ['德', 'I-Shiyi'], ['风', 'I-Shiyi'], ['的', 'O'], ['出', 'O'], ['生', 'O'], ['日', 'O'], ['期', 'O'], ['是', 'O']]
    :return:
    """
    new_word_flag = []
    words_len = len(word_flag)
    for _index, (word, flag) in enumerate(word_flag, 1):
        if flag[0] in ['B', 'O']:
            if flag[0] == 'B' and (_index == words_len or word_flag[_index][1][0] == 'O'):
                # 最后，或者独立成词
                flag = 'S' + flag[1:]
            new_word_flag.append([word, flag])
        elif flag[0] == 'I':
            if _index == words_len or word_flag[_index][1][0] == 'O':
                flag = 'E' + flag[1:]
            new_word_flag.append([word, flag])
        else:
            new_word_flag.append([word, flag])
    return new_word_flag

def main():
    TRAIN_DATA_PATH = "../data/ner_rel_train.txt"
    DEV_DATA_PATH = "../data/ner_rel_dev.txt"
    TEST_DATA_PATH = "../data/ner_rel_test.txt"

    for bio_file_name in [TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH]:
        bioes_file_name = bio_file_name[:-4] + '_BIOES.txt'

        with open(bioes_file_name, 'w')as f2:
            for word_flag, label in generator_load_data(bio_file_name):
                word_flag = bio2bioes(word_flag)
                f2.write('{}\t{}\n'.format(' '.join('/'.join(w) for w in word_flag), label))

if __name__ == '__main__':
    main()