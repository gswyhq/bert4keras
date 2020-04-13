#!/usr/bin/python3
# coding: utf-8

import random

def multi_to_one(file_name, save_file):
    with open(file_name, errors='ignore')as f:
        datas = f.read()
        datas = datas.split('\n\n')
        datas = [t.split('\n') for t in datas if t]

    rel_labels = ['类别1', '类别2', '类别3', '类别4']
    with open(save_file, 'w')as f:
        for word_flag in datas:
            f.write('{}\t{}\n'.format(' '.join(['/'.join([wf[0], wf[2:]]) for wf in word_flag if len(wf) >= 3 and wf[1]=='\t']), random.choice(rel_labels)))

def main():
    train_file = '/home/gswyhq/github_projects/ChineseNLPCorpus/NER/MSRA/zh-NER-TF/train_data'
    dev_file = '/home/gswyhq/github_projects/ChineseNLPCorpus/NER/MSRA/zh-NER-TF/test_data'

    multi_to_one(train_file, './data/ner_rel0_train_data.txt')
    multi_to_one(dev_file, './data/ner_rel0_dev_data.txt')


if __name__ == '__main__':
    main()