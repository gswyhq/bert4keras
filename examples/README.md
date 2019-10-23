
# 基于albert的情感分析
* 训练  
root@693bafbd7758:/notebooks/bert4keras# nohup python3 examples/task_sentiment_albert.py train > train.log &
* 预测  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_sentiment_albert.py 这只铅笔看起来比较好

# 基于BiLSTM+CRF的命名实体识别
* 训练  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_ner.py train
* 预测  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_ner.py 北京与上海的距离是多少？

# 基于albert的命名实体识别
* 训练  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_ner_albert.py train
* 预测  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_ner_albert.py 北京天安门在哪里

# 基于albert的意图或主题多分类
* 训练  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_classify_albert.py train
* 预测  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_classify_albert.py 养很多花是什么体验

# 基于albert的语义相似性判断
* 训练  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_similarity_albert.py train
* 预测  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/task_similarity_albert.py 今天天气很好 今天天气不错

# 基于BiLSTM命名实体识别+实体关系属性识别
* 训练  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/ner_classify.py train
* 预测  
gswyhq@gswyhq-PC:~/github_projects/bert4keras$ python3 examples/ner_classify.py 太平洋面积有多大




