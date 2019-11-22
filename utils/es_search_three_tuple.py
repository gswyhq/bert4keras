#!/usr/bin/python3
# coding: utf-8

import json
import requests
import traceback
from requests.exceptions import ConnectionError

from config import ES_PASSWORD, ES_USER, ES_PORT, ES_HOST, ES_TIMEOUT

def get_ngrams(query, n=3):
    tempQuery = str(query)
    ngrams = []
    for i in range(0, len(tempQuery)-n):
        ngrams.append(tempQuery[i:i+n])
    return ngrams

def generator_body(query_question, size=20):

    should_list = [{
                        "match": {
                            "query_question": {
                                "boost": 1,
                                "query": word
                            }
                        }
                    } for word in query_question
    ]
    must_should_list = [{
                        "term": {
                            "subject.keyword": word
                            }
                        } for word in sum([get_ngrams(query_question, n=n) for n in range(2, len(query_question))], [])]
    body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "query_question": query_question
                        }
                    }
                ] + should_list,
                # "minimum_should_match": -1
                "must": {
                        "bool": {
                            "should": must_should_list
                        }
                }
            }
        }
    }
    # print(json.dumps(body, ensure_ascii=False))
    return body

def get_random_chat_question(es_host=ES_HOST, es_port=ES_PORT, es_user=ES_USER, es_password=ES_PASSWORD):
    """
    随机搜索一条闲聊语料
    :return:
    """
    _index = 'chat_raw_chat_corpus_alias'
    body = {
              "query": {
                "function_score": {
                  "query": {
                    "match_all": {}
                  },
                  "random_score": {}
                }
              },
              "size": 1,
                "_source": {
                    "includes": [
                        "question",
                        "answer"
                    ]
                }
            }

    url = "http://{}:{}/{}/_search".format(es_host, es_port, _index)

    ret = requests.get(url=url, json=body, auth=(es_user, es_password), timeout=ES_TIMEOUT)
    result = ret.json()
    # print(result)
    rets = [t['_source'] for t in result.get('hits', {}).get('hits', [])]
    # print(rets)  # [{'question': '孙楠也就头发颜色好看。可以考虑年后染一个。哈哈', 'answer': '孙楠一上来的刹那，我以为是韩红呢！就忽略了他头发的颜色！'}]
    return rets

def search_three_tuple(question='', es_host=ES_HOST, es_port=ES_PORT, es_user=ES_USER, es_password=ES_PASSWORD):
    """
    自然语言句子，转换成查询语句，查询得到可能的三元组
    :param question:
    :param es_host:
    :param es_port:
    :param es_user:
    :param es_password:
    :return:
    """
    # tongyonggraph_three_tuple_20191101_183634
    _index = 'tongyonggraph_three_tuple_alias'
    url = "http://{}:{}/{}/_search".format(es_host, es_port, _index)
    body = generator_body(question, size=20)

    ret = requests.get(url=url, json=body, auth=(es_user, es_password), timeout=ES_TIMEOUT)
    result = ret.json()
    # print(result)
    rets = [t['_source'] for t in result.get('hits', {}).get('hits', [])]
    return [[t['subject'], t['predicate'], t['object']] for t in rets]

def try_search_three_tuple(question='', es_host=ES_HOST, es_port=ES_PORT, es_user=ES_USER, es_password=ES_PASSWORD):
    for try_i in range(3):
        try:
            rets = search_three_tuple(question=question, es_host=es_host, es_port=es_port, es_user=es_user, es_password=es_password)
            return rets
        except ConnectionError as e:
            print("请求出错：{}， 错误详情：{}".format(e, traceback.format_exc()))
            print('`{}`请求出错，再次尝试{}'.format(question, try_i))
        except Exception as e:
            print("请求出错：{}， 错误详情：{}".format(e, traceback.format_exc()))
            print('`{}`请求出错，再次尝试{}'.format(question, try_i))
            return []
    return []
def main():

    question = '“”万里长城有多长'
    ret = search_three_tuple(question=question, es_host=ES_HOST, es_port=ES_PORT, es_user=ES_USER, es_password=ES_PASSWORD)
    print(ret)

if __name__ == '__main__':
    main()