# Author:yifan
import  numpy as np
import tensorflow as tf
import parameter_config

# 构建模型  3 Transformer模型
# 生成位置嵌入
def fixedPositionEmbedding(batchSize, sequenceLen):
    embeddedPosition = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):  #类似one-hot方式的构造
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)
    return np.array(embeddedPosition, dtype="float32")

# 模型构建
class Transformer(object):
    """
    Transformer Encoder 用于文本分类
    """
    def __init__(self, config, wordEmbedding):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.embeddedPosition = tf.placeholder(tf.float32, [None, config.sequenceLength, config.sequenceLength], name="embeddedPosition")
        self.config = config
        # 定义l2损失
        l2Loss = tf.constant(0.0)
        
        # 词嵌入层, 位置向量的定义方式有两种：一是直接用固定的one-hot的形式传入，然后和词向量拼接，
        # 在当前的数据集上表现效果更好。另一种
        # 就是按照论文中的方法实现，这样的效果反而更差，可能是增大了模型的复杂度，在小数据集上表现不佳。
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embedded = tf.nn.embedding_lookup(self.W, self.inputX)
            self.embeddedWords = tf.concat([self.embedded, self.embeddedPosition], -1)

        with tf.name_scope("transformer"):
            for i in range(config.model.numBlocks):  #numBlocks==1，文本只设置了一个
                with tf.name_scope("transformer-{}".format(i + 1)):
                    # 维度[batch_size, sequence_length, embedding_size]
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.inputX, queries=self.embeddedWords,
                                                            keys=self.embeddedWords)
                    # 维度[batch_size, sequence_length, embedding_size]
                    self.embeddedWords = self._feedForward(multiHeadAtt, 
                                                           [config.model.filters, config.model.embeddingSize + config.sequenceLength])
                
            outputs = tf.reshape(self.embeddedWords, [-1, config.sequenceLength * (config.model.embeddingSize + config.sequenceLength)])
        outputSize = outputs.get_shape()[-1].value
        
        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropoutKeepProb)
    
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(outputs, outputW, outputB, name="logits")
            
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
        
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            
            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(tf.reshape(self.inputY, [-1, 1]), 
                                                                                                    dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
                
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss
            
    def _layerNormalization(self, inputs, scope="layerNorm"):
        # LayerNorm层和BN层有所不同
        epsilon = self.config.model.epsilon
        inputsShape = inputs.get_shape() # [batch_size, sequence_length, embedding_size]
        paramsShape = inputsShape[-1:]
        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(paramsShape))
        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

        return outputs
            
    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):
        # rawKeys 的作用是为了计算mask时用的，因为keys是加上了position embedding的，其中不存在padding为0的值
        numHeads = self.config.model.numHeads   #头数，目前设置为8
        keepProp = self.config.model.keepProp   #dropout数量
        
        if numUnits is None: #若是没传入值，直接去输入数据的最后一维，即embedding size.embeddingSize = 200
            numUnits = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0) 
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0) 
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # queryies = keys，因此只要一方为0，计算出的权重就为0。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        keyMasks = tf.tile(rawKeys, [numHeads, 1]) 

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings, scaledSimilary) # 维度[batch_size * numHeads, queries_len, key_len]

        # 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
        # Decoder是生成模型，主要用在语言生成中
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings, maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(maskedSimilary)

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)
        
        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        # 在这里的前向传播采用卷积神经网络
        
        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs
    
    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize
        
        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i-i%2) / embeddingSize) for i in range(embeddingSize)] 
                                      for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded