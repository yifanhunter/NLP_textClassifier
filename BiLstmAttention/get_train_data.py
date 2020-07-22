# Author:yifan
import json
from collections import Counter
import gensim
import pandas as pd
import numpy as np
import parameter_config

# 2 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate
        self._stopWordDict = {}
        self.trainReviews = []
        self.trainLabels = []
        self.evalReviews = []
        self.evalLabels = []
        self.wordEmbedding = None
        self.labelList = []
    def _readData(self, filePath):
        """
        从csv文件中读取数据集,就本次测试的文件做记录
        """
        df = pd.read_csv(filePath)  #读取文件，是三列的数据，第一列是review,第二列sentiment，第三列rate
        if self.config.numClasses == 1:
            labels = df["sentiment"].tolist()  #读取sentiment列的数据，  显示输出01序列数组25000条
        elif self.config.numClasses > 1:
            labels = df["rate"].tolist()   #因为numClasses控制，本次取样没有取超过二分类  该处没有输出
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]  #按空格语句切分
        return reviews, labels
    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[label] for label in labels]   #print(labels==labelIds) 结果显示为true,也就是两个一样
        return labelIds
    def _wordToIndex(self, reviews, word2idx):
        """将词转换成索引"""
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        # print(max(max(reviewIds)))
        # print(reviewIds)
        return reviewIds  #返回25000个无序的数组
    def _genTrainEvalData(self, x, y, word2idx, rate):
        """生成训练集和验证集 """
        reviews = []
        # print(self._sequenceLength)
        # print(len(x))
        for review in x:   #self._sequenceLength为200，表示长的切成200，短的补齐，x数据依旧是25000
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))
                # print(len(review + [word2idx["PAD"]] * (self._sequenceLength - len(review))))
        #以下是按照rate比例切分训练和测试数据：
        trainIndex = int(len(x) * rate)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")
        return trainReviews, trainLabels, evalReviews, evalLabels

    def _getWordEmbedding(self, words):
        """按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        反馈词和对应的向量（200维度），另外前面增加PAD对用0的数组，UNK对应随机数组。
        """
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []
        # 添加 "pad" 和 "UNK",
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))  # _embeddingSize 本文定义的是200
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        # print(wordEmbedding)
        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")
        # print(vocab[:3],wordEmbedding[:3])
        return vocab, np.array(wordEmbedding)
    def _genVocabulary(self, reviews, labels):
        """生成词向量和词汇-索引映射字典，可以用全数据集"""
        allWords = [word for review in reviews for word in review]   #单词数量5738236   reviews是25000个观点句子【】
        subWords = [word for word in allWords if word not in self.stopWordDict]   # 去掉停用词
        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True) #返回键值对，并按照数量排序
        # print(len(sortWordCount))  #161330
        # print(sortWordCount[:4],sortWordCount[-4:]) # [('movie', 41104), ('film', 36981), ('one', 24966), ('like', 19490)] [('daeseleires', 1), ('nice310', 1), ('shortsightedness', 1), ('unfairness', 1)]
        words = [item[0] for item in sortWordCount if item[1] >= 5]   # 去除低频词，低于5的
        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        word2idx = dict(zip(vocab, list(range(len(vocab)))))   #生成类似这种{'I': 0, 'love': 1, 'yanzi': 2}
        uniqueLabel = list(set(labels))    #标签去重  最后就 0  1了
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel))))) #本文就 {0: 0, 1: 1}
        self.labelList = list(range(len(uniqueLabel)))
        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        with open("../data/wordJson/label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f)
        return word2idx, label2idx

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 初始化停用词
        self._readStopWord(self._stopWordSource)
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)
        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)
        # 将标签和句子数值化
        labelIds = self._labelToIndex(labels, label2idx)
        reviewIds = self._wordToIndex(reviews, word2idx)
        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labelIds, word2idx,
                                                                                    self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

#获取前些模块的数据
# config =parameter_config.Config()
# data = Dataset(config)
# data.dataGen()