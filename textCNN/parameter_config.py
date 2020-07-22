# Author:yifan
#需要的所有导入包，存放留用，转换到jupyter后直接使用
# 1 配置训练参数
class TrainingConfig(object):
    epoches = 5
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 200
    numFilters = 128
    filterSizes = [2, 3, 4, 5]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

#上面两个都放到Config一个类中，为了后期调用方便
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