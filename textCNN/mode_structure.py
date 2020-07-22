# Author:yifan

import tensorflow as tf
import parameter_config

# 构建模型  3 textCNN 模型
class TextCNN(object):
    """
    Text CNN 用于文本分类
    """
    def __init__(self, config, wordEmbedding):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX") #占据【不定，200】的矩阵
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
          #词对应的向量wordEmbedding（200维度），另外前面增加PAD对用0的数组，UNK对应随机数组。
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
            # 卷积的输入是四维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度（通道数）
            self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)

        # 创建卷积和池化层
        pooledOutputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filterSize in enumerate(config.model.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                # 卷积层，卷积核尺寸为 filterSize * embeddingSize，卷积核的个数为numFilters    filterSizes  = [2, 3, 4, 5]
                # 初始化权重矩阵和偏置
                filterShape = [filterSize, config.model.embeddingSize, 1, config.model.numFilters]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
                conv = tf.nn.conv2d(
                    self.embeddedWordsExpanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.sequenceLength - filterSize + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度
        numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)   #128*filterSizes的个数

        # 池化后的维度不变，按照最后的维度channel来concat
        self.hPool = tf.concat(pooledOutputs, 3)

        # 摊平成二维的数据输入到全连接层
        self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])

        # dropout
        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)

        # 全连接层的输出
        with tf.name_scope("output"):   #predice.py的graph.get_tensor_by_name("output/predictions:0")用到调用
            outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")#≥0.0则输出1
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
            print(self.predictions)

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,   #二分类用sigmoid
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                 dtype=tf.float32))
            elif config.numClasses > 1:
                #多分类使用softmax
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss