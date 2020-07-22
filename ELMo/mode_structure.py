# Author:yifan
# _*_ coding:utf-8 _*_
import tensorflow as tf
import parameter_config

config = parameter_config.Config()
# 构建模型  3 ELMo模型
# 构建模型
class ELMo(object):
    """"""
    def __init__(self, config):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.float32, [None, config.sequenceLength, config.model.embeddingSize], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        
        # 定义l2损失
        l2Loss = tf.constant(0.0)
        
        with tf.name_scope("embedding"):
            embeddingW = tf.get_variable(
                "embeddingW",
                shape=[config.model.embeddingSize, config.model.embeddingSize],
                initializer=tf.contrib.layers.xavier_initializer())
            reshapeInputX = tf.reshape(self.inputX, shape=[-1, config.model.embeddingSize])
            
            self.embeddedWords = tf.reshape(tf.matmul(reshapeInputX, embeddingW), shape=[-1, config.sequenceLength, config.model.embeddingSize])
            self.embeddedWords = tf.nn.dropout(self.embeddedWords, self.dropoutKeepProb)
            
        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                                 output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                                 output_keep_prob=self.dropoutKeepProb)


                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元组(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                                  self.embeddedWords, dtype=tf.float32,
                                                                                  scope="bi-lstm" + str(idx))
        
                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords = tf.concat(outputs_, 2)
        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embeddedWords, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]
            # 得到Attention的输出
            output = self._attention(H)
            outputSize = config.model.hiddenSizes[-1]
        
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.0), tf.float32, name="binaryPreds")
        
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss
    
    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = config.model.hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.sequenceLength, 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
        
        return output