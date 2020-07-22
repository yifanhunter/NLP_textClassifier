# Author:yifan

import tensorflow as tf
import parameter_config

# 3 构建模型  Bi-LSTM模型
class BiLSTM(object):
    """
    Bi-LSTM 用于文本分类
    """
    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                  self.embeddedWords, dtype=tf.float32,
                                                                                  scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    self.embeddedWords = tf.concat(outputs, 2)

        # 去除最后时间步的输出作为全连接的输入
        finalOutput = self.embeddedWords[:, 0, :]

        outputSize = config.model.hiddenSizes[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        output = tf.reshape(finalOutput, [-1, outputSize])  # reshape成全连接层的输入维度

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss