# Author:yifan
import os
import csv
import time
import datetime
import random
import json
from collections import Counter
from math import sqrt
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import parameter_config
import get_train_data
config =parameter_config.Config()
data = get_train_data.Dataset(config)
#7预测代码
# x = "this movie is full of references like mad max ii the wild one and many others the ladybug´s face it´s a clear reference or tribute to peter lorre this movie is a masterpiece we´ll talk much more about in the future"
x = "his"
# with open("../data/charJson/charToIndex.json", "w", encoding="utf-8") as f:
#     json.dump(self._charToIndex, f)
# with open("../data/charJson/indexToChar.json", "w", encoding="utf-8") as f:
#     json.dump(self._indexToChar, f)

# chars = [char for char in data._alphabet]
# vocab, charEmbedding = data._getCharEmbedding(chars)
#
# data._charToIndex = dict(zip(vocab, list(range(len(vocab)))))
# data._indexToChar = dict(zip(list(range(len(vocab))), vocab))
#
# # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
# with open("../data/charJson/charToIndex.json", "w", encoding="utf-8") as f:
#     json.dump(data._charToIndex, f)
# with open("../data/charJson/indexToChar.json", "w", encoding="utf-8") as f:
#     json.dump(data._indexToChar, f)

# reviews = []
# for i in range(len(x)):
#     reviewVec = data._reviewProcess(x[i], config.sequenceLength, data._charToIndex)
#     reviews.append(reviewVec)


# 初始化词汇-索引映射表和词向量矩阵
y = list(x)
data._genVocabulary(y)
print(x)
reviewVec = data._reviewProcess(y, config.sequenceLength, data._charToIndex)
print(reviewVec)