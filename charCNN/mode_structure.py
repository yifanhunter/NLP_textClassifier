# Author:yifan

import tensorflow as tf
import math
import parameter_config

# 构建模型  3 textCNN 模型
# 定义char-CNN分类器
class CharCNN(object):
    """
    char-CNN用于文本分类   　
    在charCNN 模型中我们引入了BN层，但是效果并不明显，甚至存在一些收敛问题，待之后去探讨。
    """
    def __init__(self, config, charEmbedding):
        # placeholders for input, output and dropuot
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.isTraining = tf.placeholder(tf.bool, name="isTraining")
        self.epsilon = config.model.epsilon
        self.decay = config.model.decay

        # 字符嵌入
        with tf.name_scope("embedding"):
            # 利用one-hot的字符向量作为初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(charEmbedding, dtype=tf.float32, name="charEmbedding"), name="W")
            # 获得字符嵌入
            self.embededChars = tf.nn.embedding_lookup(self.W, self.inputX)
            # 添加一个通道维度
            self.embededCharsExpand = tf.expand_dims(self.embededChars, -1)

        for i, cl in enumerate(config.model.convLayers):
            print("开始第" + str(i + 1) + "卷积层的处理")
            # 利用命名空间name_scope来实现变量名复用
            with tf.name_scope("convLayer-%s" % (i + 1)):
                # 获取字符的向量长度
                filterWidth = self.embededCharsExpand.get_shape()[2].value
                # filterShape = [height, width, in_channels, out_channels]
                filterShape = [cl[1], filterWidth, 1, cl[0]]
                stdv = 1 / math.sqrt(cl[0] * cl[1])

                # 初始化w和b的值
                wConv = tf.Variable(tf.random_uniform(filterShape, minval=-stdv, maxval=stdv),
                                    dtype='float32', name='w')
                bConv = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name='b')

                #                 w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
                #                 b_conv = tf.Variable(tf.constant(0.1, shape=[cl[0]]), name="b")
                # 构建卷积层，可以直接将卷积核的初始化方法传入（w_conv）
                conv = tf.nn.conv2d(self.embededCharsExpand, wConv, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # 加上偏差
                hConv = tf.nn.bias_add(conv, bConv)
                # 可以直接加上relu函数，因为tf.nn.conv2d事实上是做了一个卷积运算，然后在这个运算结果上加上偏差，再导入到relu函数中
                hConv = tf.nn.relu(hConv)

                #                 with tf.name_scope("batchNormalization"):
                #                     hConvBN = self._batchNorm(hConv)

                if cl[-1] is not None:
                    ksizeShape = [1, cl[2], 1, 1]
                    hPool = tf.nn.max_pool(hConv, ksize=ksizeShape, strides=ksizeShape, padding="VALID", name="pool")
                else:
                    hPool = hConv

                print(hPool.shape)

                # 对维度进行转换，转换成卷积层的输入维度
                self.embededCharsExpand = tf.transpose(hPool, [0, 1, 3, 2], name="transpose")
        print(self.embededCharsExpand)
        with tf.name_scope("reshape"):
            fcDim = self.embededCharsExpand.get_shape()[1].value * self.embededCharsExpand.get_shape()[2].value
            self.inputReshape = tf.reshape(self.embededCharsExpand, [-1, fcDim])

        weights = [fcDim] + config.model.fcLayers

        for i, fl in enumerate(config.model.fcLayers):   #fcLayers = [512]
            with tf.name_scope("fcLayer-%s" % (i + 1)):
                print("开始第" + str(i + 1) + "全连接层的处理")
                stdv = 1 / math.sqrt(weights[i])
                # 定义全连接层的初始化方法，均匀分布初始化w和b的值
                wFc = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv), dtype="float32",
                                  name="w")
                bFc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype="float32", name="b")

                #                 w_fc = tf.Variable(tf.truncated_normal([weights[i], fl], stddev=0.05), name="W")
                #                 b_fc = tf.Variable(tf.constant(0.1, shape=[fl]), name="b")

                self.fcInput = tf.nn.relu(tf.matmul(self.inputReshape, wFc) + bFc)
                with tf.name_scope("dropOut"):
                    self.fcInputDrop = tf.nn.dropout(self.fcInput, self.dropoutKeepProb)
            self.inputReshape = self.fcInputDrop

        with tf.name_scope("outputLayer"):
            stdv = 1 / math.sqrt(weights[-1])
            # 定义隐层到输出层的权重系数和偏差的初始化方法
            #             w_out = tf.Variable(tf.truncated_normal([fc_layers[-1], num_classes], stddev=0.1), name="W")
            #             b_out = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            wOut = tf.Variable(tf.random_uniform([config.model.fcLayers[-1], 1], minval=-stdv, maxval=stdv),
                               dtype="float32", name="w")
            bOut = tf.Variable(tf.random_uniform(shape=[1], minval=-stdv, maxval=stdv), name="b")
            # tf.nn.xw_plus_b就是x和w的乘积加上b
            self.predictions = tf.nn.xw_plus_b(self.inputReshape, wOut, bOut, name="predictions")
            # 进行二元分类
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.0), tf.float32, name="binaryPreds")

        with tf.name_scope("loss"):
            # 定义损失函数，对预测值进行softmax，再求交叉熵。
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            self.loss = tf.reduce_mean(losses)

    def _batchNorm(self, x):
        # BN层代码实现
        gamma = tf.Variable(tf.ones([x.get_shape()[3].value]))
        beta = tf.Variable(tf.zeros([x.get_shape()[3].value]))
        self.popMean = tf.Variable(tf.zeros([x.get_shape()[3].value]), trainable=False, name="popMean")
        self.popVariance = tf.Variable(tf.ones([x.get_shape()[3].value]), trainable=False, name="popVariance")

        def batchNormTraining():
            # 一定要使用正确的维度确保计算的是每个特征图上的平均值和方差而不是整个网络节点上的统计分布值
            batchMean, batchVariance = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
            decay = 0.99
            trainMean = tf.assign(self.popMean, self.popMean * self.decay + batchMean * (1 - self.decay))
            trainVariance = tf.assign(self.popVariance,
                                      self.popVariance * self.decay + batchVariance * (1 - self.decay))

            with tf.control_dependencies([trainMean, trainVariance]):
                return tf.nn.batch_normalization(x, batchMean, batchVariance, beta, gamma, self.epsilon)

        def batchNormInference():
            return tf.nn.batch_normalization(x, self.popMean, self.popVariance, beta, gamma, self.epsilon)
        batchNormalizedOutput = tf.cond(self.isTraining, batchNormTraining, batchNormInference)
        return tf.nn.relu(batchNormalizedOutput)