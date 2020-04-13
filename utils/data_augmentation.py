#!/usr/bin/python3
# coding: utf-8

import random
import json
import copy
import re
import jieba
from gensim.models import KeyedVectors

def random_weight(weight_data, key_list=None, total=None):
    """
    根据字符串的权重，随机选择
    :param weight_data: {'a': 10, 'b': 2, 'c': 1}
    :param key_list: 随机选择的顺序表
    :param total: 所有权重和
    :return:
    """
    if total is None:
        total = sum(weight_data.values())    # 权重求和

    ra = random.uniform(0, total)   # 在0与权重和之前获取一个随机数
    curr_sum = 0
    ret = None
    if key_list is None:
        key_list = list(weight_data.keys())
        key_list.sort()
    for k in key_list:
        curr_sum += weight_data[k]             # 在遍历中，累加当前权重值
        if ra <= curr_sum:          # 当随机数<=当前权重和时，返回权重key
            ret = k
            break
    return ret

def generator_bio_format(text, entity_dict):
    """
    根据输入的文本及实体词典，生成对应的bio格式数据
    :param text:
    :param entity_dict:
    :return:
    """
    word_flag = []
    if entity_dict and any(key in text for key in entity_dict.keys()):
        index_list = [[i.start(), i.end()] for i in re.finditer('|'.join(list(entity_dict.keys())), text)]
        for _index, char in enumerate(text):
            if any(start_index < _index < end_index for start_index, end_index in index_list):
                continue

            word = ''
            for start_index, end_index in index_list:
                if _index == start_index:
                    word = text[start_index: end_index]
                    break
                elif _index < start_index:
                    break
                else:
                    continue

            if word:
                word = word.upper()
                word_len = len(word)
                word_pinyin = entity_dict.get(word, 'Shiyi')
                if word_len == 1:
                    word_flag.append([word, 'B-{}'.format(word_pinyin)])
                elif word_len >= 2:
                    word_flag.append([word[0], 'B-{}'.format(word_pinyin)])
                    for w in word[1:]:
                        word_flag.append([w, 'I-{}'.format(word_pinyin)])
                else:
                    continue
            else:
                word_flag.append([char, 'O'])
    else:
        [word_flag.append([char, 'O']) for char in text]
    # print(''.join([word for word, flag in word_flag]), ''.join([flag for word, flag in word_flag]))
    return word_flag

def random_insert(origin_words, choice_word, similar_word, entity_dict, words_weight_data, words_key_list, words_total):
    random_num = random.random()
    if random_num <= 0.33:
        insert_index = 0
    elif random_num <= 0.66:
        insert_index = len(origin_words)
    else:
        # 若插入在居中，则随机选择一个插入位
        insert_index = random.randint(0, len(origin_words))
    random_word = random_weight(words_weight_data, key_list=words_key_list, total=words_total)
    insert_text = ''.join(origin_words[:insert_index] + [random_word] + origin_words[insert_index:])
    word_flag = generator_bio_format(insert_text.replace(choice_word, similar_word), entity_dict)
    return word_flag

def generator_train_data_augmentation(text, entity_dict, intent, wv_from_text, words_weight_data, words_key_list, words_total):
    """
    根据输入的标注语料生成训练及验证数据
    生成训练语料方法，同义词替换，随机插入字符；
    :param text:
    :param entity_dict:
    :param intent:
    :return:
    """
    # print(text, entity_dict, intent)
    train_data, validation_data = [], []
    [jieba.add_word(word, freq=2000, tag=flag) for word, flag in entity_dict.items()]
    origin_words = [word for word in jieba.cut(text)]

    origin_word_flag = generator_bio_format(text, entity_dict)
    validation_data.append([origin_word_flag, intent])
    current_entity_dict = copy.deepcopy(entity_dict)
    current_words = copy.deepcopy(origin_words)

    # 句子越长，变换次数越多
    for _ in range(int(len(origin_words)/3)):
        # 变换顺序
        random.shuffle(current_words)
        word_flag = generator_bio_format(''.join(current_words), entity_dict)
        train_data.append([word_flag, intent])
        random.shuffle(current_words)
        word_flag = generator_bio_format(''.join(current_words), entity_dict)
        validation_data.append([word_flag, intent])

    for _ in range(min(len(origin_words), 10)):

        # 随机插入
        word_flag = random_insert(origin_words, '', '', entity_dict, words_weight_data,
                                  words_key_list, words_total)
        train_data.append([word_flag, intent])

        word_flag = random_insert(origin_words, '', '', entity_dict, words_weight_data,
                                  words_key_list, words_total)
        validation_data.append([word_flag, intent])

        # 随机选择一个词，拟进行同义词替换
        choice_word = random.choice(origin_words)
        if not choice_word in wv_from_text.index2word:
            continue
        for similar_word, score in wv_from_text.similar_by_word(choice_word):
            if score < 0.85:
                break

            if entity_dict.get(choice_word):
                current_entity_dict[similar_word] = entity_dict.get(choice_word)

            # 同义词替换
            word_flag = generator_bio_format(text.replace(choice_word, similar_word), current_entity_dict)
            train_data.append([word_flag, intent])

            # 同义词替换+随机插入
            word_flag = random_insert(origin_words, choice_word, similar_word, current_entity_dict, words_weight_data, words_key_list, words_total)
            train_data.append([word_flag, intent])

    # print(train_data, validation_data)
    return train_data, validation_data, origin_word_flag



def main():
    with open('/home/gswyhq/data/微博词频统计.json')as f:
        words_weight_data = json.load(f)
    words_key_list = list(words_weight_data.keys())
    words_total = sum(words_weight_data.values())
    # random_word = random_weight(words_weight_data, key_list=words_key_list, total=words_total)
    # print(random_word)
    wv_from_text = KeyedVectors.load_word2vec_format('/home/gswyhq/data/WordVector_60dimensional/wiki.zh.embedding.txt', binary=False)
    text = "地中海贫血症状是啥"
    entity_dict = {"地中海贫血": "Shiyi"}
    intent = "症状"
    train_data, validation_data, origin_word_flag = generator_train_data_augmentation(text, entity_dict, intent, wv_from_text, words_weight_data, words_key_list, words_total)
    print(train_data)
    print(validation_data)

if __name__ == '__main__':
    main()