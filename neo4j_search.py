#!/usr/bin/python3
# coding: utf-8

import os
import traceback
import re
import requests
from pypinyin import pinyin, lazy_pinyin

NEO4J_HOST = '192.168.3.133'
NEO4J_HTTP_PORT = 7476
NEO4J_USER = 'neo4j' 
NEO4J_PASSWORD = 'gswyhq'

def get_pinyin(text, upper=False):
    """
    返回文本的拼音，若upper为真则所有字母都大写，否则仅仅首字母大写
    :param text: 文本
    :return: Shuxing
    """
    if isinstance(text, str):
        if upper:
            return ''.join(lazy_pinyin(text)).upper()
        else:
            return ''.join(lazy_pinyin(text)).capitalize()
    return ''

def post_statements(statements):
    '''
    {
      "statements" : [ {
        "statement" : "CREATE (n) RETURN id(n)"
      }, {
        "statement" : "CREATE (n {props}) RETURN n",
        "parameters" : {
          "props" : {
            "name" : "My Node"
          }
        }
      },
    {
        "statement" : "match (n ) where n.name = {props}.name RETURN n",
        "parameters" : {
          "props" : {
            "name" : "恶性肿瘤"
          }
        }
      }
     ]
    :param Config:
    :param statements:
    :return:
    '''
    url = 'http://{host}:{http_port}/db/data/transaction/commit'.format(host=NEO4J_HOST,
                                                                        http_port=NEO4J_HTTP_PORT)
    body = {
        "statements": statements
    }
    # print('批量写入cypher数量：{}'.format(len(statements)))
    # print("body = ", body)
    # print("url: {}, dict: {}".format(url, [Config.NEO4J_USER,  Config.NEO4J_PASSWORD]))
    # print("""curl {} -u neo4j:gswyhq -H "Content-Type: application/json; charset=UTF-8" -d '{}' """.format(url, json.dumps(body, ensure_ascii=0)))
    r = requests.post(url, json=body, headers={"Content-Type": "application/json; charset=UTF-8","Connection":"close"},
                      auth=(NEO4J_USER, NEO4J_PASSWORD), timeout=(60 * 120, 60 * 120))
    ret = r.json()
    errors = ret.get("errors", [])
    if errors:
        print("向neo4j写入出错：{}".format(errors))
    assert not errors, '【错误】neo4j数据写入出错！'
    return ret

def search_entity(entity_dict, relationship):
    """
    根据实体类型及其属性，随机搜索一个同类的实体；
    :param entity_dict:
    :param relationship:
    :return:
    """
    statements = []
    entity_names = []
    entity_replace_dict = {}
    for entity_name, entity_type in entity_dict.items():
        statement = {
            "statement": '''MATCH (n: {})-[:`{}`]->(m) where n.name <> "{}" and id(n) > rand() * 13078375 RETURN DISTINCT n.name limit 1 '''.format(get_pinyin(entity_type), relationship, entity_name)
        }
        statements.append(statement)
        entity_names.append(entity_name)
        # print(statements)
    ret = post_statements(statements)
    # print(ret)
    # {'results': [{'columns': ['n.name'], 'data': [{'row': ['山歌民谣'], 'meta': [None]}]}, {'columns': ['n.name'], 'data': [{'row': ['风险中和'], 'meta': [None]}]}], 'errors': []}

    for entity_name, row_data in zip(entity_names, ret.get('results', [])):
        # print(row_data)
        if row_data.get('data') and row_data['data'][0].get('row'):
            new_entity_name = row_data['data'][0]['row'][0]
            entity_replace_dict.setdefault(entity_name, new_entity_name)
        entity_replace_dict.setdefault(entity_name, entity_name)
    # print(entity_replace_dict)
    return entity_replace_dict

def main():
    entity_dict = {"同安县": 'Tongyong', '太平洋': 'Shuyu'}
    rel = '概念'
    search_entity(entity_dict, rel)

if __name__ == '__main__':
    main()