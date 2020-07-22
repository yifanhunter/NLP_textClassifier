# Author:yifan
import json
import pandas as pd
import  numpy as np
import parameter_config
# 2、 训练数据生成
# 　　1） 加载数据，将所有的句子分割成字符表示
# 　　2） 构建字符-索引映射表，并保存成json的数据格式，方便在inference阶段加载使用
# 　　3）将字符转换成one-hot的嵌入形式，作为模型中embedding层的初始化值。
# 　　4） 将数据集分割成训练集和验证集
# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):   #config.的部分都是从parameter.config.py中带出
        self._dataSource = config.dataSource             #路径
        self._sequenceLength = config.sequenceLength    # 字符表示的序列长度
        self._rate = config.rate                        # 训练集的比例
        self._alphabet = config.alphabet
        self.trainReviews = []
        self.trainLabels = []
        self.evalReviews = []
        self.evalLabels = []
        self.charEmbedding = None
        self._charToIndex = {}
        self._indexToChar = {}

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["review"].tolist()
        reviews = [[char for char in line if char != " "] for line in review]
        return reviews, labels

    def _reviewProcess(self, review, sequenceLength, charToIndex):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """
        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength
        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)
        for i in range(sequenceLen):
            if review[i] in charToIndex:
                reviewVec[i] = charToIndex[review[i]]
            else:
                reviewVec[i] = charToIndex["UNK"]
        return reviewVec

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集,最后生成的一行表示一个句子，包含单词数为sequenceLength = 1014。每个单词用index表示
        """
        reviews = []
        labels = []
        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._charToIndex)
            reviews.append(reviewVec)
            labels.append([y[i]])
        trainIndex = int(len(x) * rate)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")
        return trainReviews, trainLabels, evalReviews, evalLabels

    def _getCharEmbedding(self, chars):
        """
        按照one的形式将字符映射成向量
        字母pad表示【0，0，0...】,UNK是【1，0，0...】，a表示【0，1，0...】等等
        """
        alphabet = ["UNK"] + [char for char in self._alphabet]
        vocab = ["pad"] + alphabet
        charEmbedding = []
        charEmbedding.append(np.zeros(len(alphabet), dtype="float32"))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype="float32")
            # 生成每个字符对应的向量
            onehot[i] = 1
            # 生成字符嵌入的向量矩阵
            charEmbedding.append(onehot)
        return vocab, np.array(charEmbedding)

    def _genVocabulary(self, reviews):
        """
        生成字符向量和字符-索引映射字典
        """
        chars = [char for char in self._alphabet]
        vocab, charEmbedding = self._getCharEmbedding(chars)
        self.charEmbedding = charEmbedding

        self._charToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToChar = dict(zip(list(range(len(vocab))), vocab))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/charJson/charToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._charToIndex, f)
        with open("../data/charJson/indexToChar.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToChar, f)

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 初始化数据集
        # reviews: [['"', 'w', 'i', 't', 'h', 'a', 'l', 'l', 't', 'h', 'i', 's', 's', 't', 'u', 'f', 'f
        #labels:[1, ...
        reviews, labels = self._readData(self._dataSource)
        # print(reviews[0])    #['"', 'w', 'i', 't', 'h', 'a', 'l', 'l', 't', 'h', 'i',
        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)
        # print(reviews[0])   #['"', 'w', 'i', 't', 'h', 'a', 'l', 'l', 't', 'h', 'i', 's', 's', 't', 'u', 'f', 'f', 'g', 'o', 'i'
        # 初始化训练集和测试集  训练集20000，测试集5000   每个trainReviews 长度位1014
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        # print("++++++++++")
        # print(trainReviews[0])   #[46 24 10 ...  6  5 17]
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels
        # print(trainReviews)
        # print("++++")
        # print(trainLabels)
        # print(len(trainReviews[0]))
        # print(len(trainReviews[2]))
        # print(len(evalLabels))
#test
config =parameter_config.Config()
data = Dataset(config)
data.dataGen()