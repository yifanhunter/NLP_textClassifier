# Author:yifan
# _*_ coding:utf-8 _*_
#需要的所有导入包，存放留用，转换到jupyter后直接使用
# 1 配置训练参数

class TrainingConfig(object):
    epoches = 5
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001
    
class ModelConfig(object):
    embeddingSize = 256  # 这个值是和ELMo模型的output Size 对应的值
    
    hiddenSizes = [128]  # LSTM结构的神经元个数
    
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0
    
class Config(object):
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 128
    
    dataSource = "../data/preProcess/labeledTrain.csv"
    
    stopWordSource = "../data/english"
    
    optionFile = "../data/elmodata/elmo_options.json"
    weightFile = "../data/elmodata/elmo_weights.hdf5"
    vocabFile = "../data/elmodata/vocab.txt"
    tokenEmbeddingFile = '../data/elmodata/elmo_token_embeddings.hdf5'
    
    numClasses = 2
    
    rate = 0.8  # 训练集的比例
    
    training = TrainingConfig()
    
    model = ModelConfig()

    
# 实例化配置参数对象
# config = Config()