# 配置参数
class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 200
    hiddenSizes = [256, 256]  # 单层LSTM结构的神经元个数
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 128
    dataSource = "../data/preProcess/labeledTrain.csv"
    stopWordSource = "../data/english"
    numClasses = 1  # 二分类设置为1，多分类设置为类别的数目
    rate = 0.8  # 训练集的比例
    training = TrainingConfig()
    model = ModelConfig()

# 实例化配置参数对象
# config = Config()