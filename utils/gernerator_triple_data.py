#!/usr/bin/python3
# coding: utf-8

import os
import sys
import re
import json
import jieba
from jieba import posseg

sys.path.append('.')

from utils.es_search_three_tuple import try_search_three_tuple
from examples.ner_classify_albert_tiny import generator_bio_format

replace_dict = {'yiwai': '摔伤',
             'zhiye': '教师',
             'shiyi': '攀岩',
             'jibing': '恶性肿瘤',
             'didian': '北京市',
             'yiyuan': '人民医院',
             'yaopin': '狂犬疫苗',
             'zhenliao': '彩超',
             'changjing': '场景',
             'baoxianjihua': 'A计划',
             'mingcijieshi': '周岁',
             'baoxiangongsi': 'B保险',
             'baoquanxiangmu': '减额缴清',
             'baoxianchanpin': 'A产品',
             'baoxianzhonglei': '重疾险',
             'baozhangxiangmu': '重大疾病保险金',
             'zhengjianleixing': '居民身份证',
             'fuwuwangdianmingcheng': '第一营业部'}

def generator_triple(input_file, output_file):
    with open(input_file, 'r')as fr:
        print('开始文件：{}'.format(input_file))
        _index = 0
        with open(output_file, 'w', encoding='utf-8')as fw:
            line = fr.readline()
            while line:
                word_flag_str, rel = line.strip().rsplit('\t', maxsplit=1)
                word_flag = [word.split('/', maxsplit=1) for word in word_flag_str.split() if word[1] == '/']
                text = ''.join([word for word, flag in word_flag])
                triple = try_search_three_tuple(text)
                data = {'word_flag': word_flag,
                        'rel': rel,
                        'triple': triple}
                fw.write(json.dumps(data, ensure_ascii=False) + '\n')
                if _index % 10000 == 0:
                    print('完成：{}'.format(_index))
                line = fr.readline()
                _index += 1

def generator_baoxian_triple(input_file, output_file):
    '''保险意图语料'''

    temp_data_file = '{}.temp'.format(input_file)
    command = 'less {} |uniq > {}'.format(input_file, temp_data_file)
    os.system(command)

    with open(temp_data_file)as f:
        datas = f.readlines()

    os.system('rm {}'.format(temp_data_file))

    datas = [t.strip().rsplit('\t', maxsplit=1) for t in datas if '\t' in t]
    with open('/home/gswyhq/data/admin_bussiness_site_webot_ai_19_kg_entity_synonyms_20191106_170150.json')as f:
        kg_entity_synonyms = json.load(f)
    hits = kg_entity_synonyms.get('hits', {}).get('hits', [])
    _source = [t['_source'] for t in hits]

    for entity_type, entity in replace_dict.items():
        jieba.add_word(entity, 2000, 'Shiyi')
    for synonyms in _source:
        word = synonyms['实体标准词']
        synonyms_words = synonyms.get('实体同义词', [])
        [jieba.add_word(w, 2000, 'Shiyi') for w in synonyms_words+[word]]

    with open(output_file, 'w', encoding='utf-8')as fw:
        for text, rel in datas:
            if rel == '实体':
                continue
            text = text.lower()
            for entity_type, entity in sorted(replace_dict.items(), key=lambda x: len(x[0]), reverse=True):
                text = text.replace(entity_type, entity)
            entity_dict = {word: flag for word, flag in posseg.lcut(text) if flag == 'Shiyi'}
            word_flag = generator_bio_format(text, entity_dict)
            if entity_dict:
                subject = list(entity_dict.keys())[0]
            else:
                subject = ''
            data = {"word_flag": word_flag,
                    "rel": rel,
                    "triple": [[subject, rel, '']]}
            if subject in rel:
                continue
            fw.write(json.dumps(data, ensure_ascii=False) + '\n')



def main():
    # generator_triple('./data/ner_rel_dev_BIOES.txt.temp.aug', './data/ner_rel_dev_BIOES.txt.temp.aug.triple')
    # generator_triple('./data/ner_rel_train_BIOES.txt.temp.aug', './data/ner_rel_train_BIOES.txt.temp.aug.triple')

    generator_baoxian_triple('./data/标准项目意图训练语料2019_1019_2.txt', './data/标准项目意图训练语料2019_1019_2.txt.triple')

if __name__ == '__main__':
    main()