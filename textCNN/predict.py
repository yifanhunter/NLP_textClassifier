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
config =parameter_config.Config()

#7预测代码
x = "this movie is full of references like mad max ii the wild one and many others the ladybug´s face it´s a clear reference or tribute to peter lorre this movie is a masterpiece we´ll talk much more about in the future"
# x = "his movie is the same as the third level movie. There's no place to look good"
# x = "This film is not good"   #最终反馈为0
# x = "This film is   bad"   #最终反馈为0

# 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
with open("../data/wordJson/word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)
with open("../data/wordJson/label2idx.json", "r", encoding="utf-8") as f:   #label2idx.json内容{"0": 0, "1": 1}
    label2idx = json.load(f)
idx2label = {value: key for key, value in label2idx.items()}

#x 的处理，变成模型能识别的向量xIds
xIds = [word2idx.get(item, word2idx["UNK"]) for item in x.split(" ")]  #返回x对应的向量
if len(xIds) >= config.sequenceLength:   #xIds 句子单词个数是否超过了sequenceLength（200）
    xIds = xIds[:config.sequenceLength]
    print("ddd",xIds)
else:
    xIds = xIds + [word2idx["PAD"]] * (config.sequenceLength - len(xIds))
    print("xxx", xIds)

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # 恢复模型
        checkpoint_file = tf.train.latest_checkpoint("../model/textCNN/model/")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # 获得输出的结果
        predictions = graph.get_tensor_by_name("output/predictions:0") # mode_structure中的定义
        pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]

# print(pred)
pred = [idx2label[item] for item in pred]
print(pred)