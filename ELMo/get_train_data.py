# _*_ coding:utf-8 _*_
# Author:yifan
import json
from collections import Counter
import gensim
import pandas as pd
import numpy as np
import parameter_config

# 2 数据预处理的类，生成训练集和测试集

# 数据预处理的类，生成训练集和测试集

class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource  
        self._optionFile = config.optionFile
        self._weightFile = config.weightFile
        self._vocabFile = config.vocabFile
        self._tokenEmbeddingFile = config.tokenEmbeddingFile
        
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate
        
        self.trainReviews = []
        self.trainLabels = []
        
        self.evalReviews = []
        self.evalLabels = []
        
    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels
    
    def _genVocabFile(self, reviews):
        """
        用我们的训练数据生成一个词汇文件，并加入三个特殊字符
        """
        allWords = [word for review in reviews for word in review]
        wordCount = Counter(allWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sortWordCount.items()]
        allTokens = ['<S>', '</S>', '<UNK>'] + words
        with open(self._vocabFile, 'w',encoding='UTF-8') as fout:
            fout.write('\n'.join(allTokens))
    
    def _fixedSeq(self, reviews):
        """
        将长度超过200的截断为200的长度
        """
        return [review[:self._sequenceLength] for review in reviews]
    
    def _genElmoEmbedding(self):
        """
        调用ELMO源码中的dump_token_embeddings方法，基于字符的表示生成词的向量表示。并保存成hdf5文件，
        文件中的"embedding"键对应的value就是
        词汇表文件中各词汇的向量表示，这些词汇的向量表示之后会作为BiLM的初始化输入。
        """
        dump_token_embeddings(
            self._vocabFile, self._optionFile, self._weightFile, self._tokenEmbeddingFile)

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """
        y = [[item] for item in y]
        trainIndex = int(len(x) * rate)
        
        trainReviews = x[:trainIndex]
        trainLabels = y[:trainIndex]
        
        evalReviews = x[trainIndex:]
        evalLabels = y[trainIndex:]

        return trainReviews, trainLabels, evalReviews, evalLabels
        
            
    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)
#         self._genVocabFile(reviews) # 生成vocabFile
#         self._genElmoEmbedding()  # 生成elmo_token_embedding
        reviews = self._fixedSeq(reviews)
        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

# from data import TokenBatcher
# #获取前些模块的数据
# config =parameter_config.Config()
# data = Dataset(config)
# data.dataGen()
# batcher = TokenBatcher(config.vocabFile)
# print(batcher)