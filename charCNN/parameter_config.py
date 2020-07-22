# Author:yifan
# 1、参数配置
class TrainingConfig(object):
    epoches = 6
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    # 该列表中子列表的三个元素分别:卷积核的数量，卷积核的高度，池化的尺寸
    convLayers = [[256, 7, 4],
                  [256, 7, 4],
                  [256, 3, 4]]
    fcLayers = [512]
    dropoutKeepProb = 0.5
    epsilon = 1e-3  # BN层中防止分母为0而加入的极小值
    decay = 0.999  # BN层中用来计算滑动平均的值

class Config(object):
 # 我们使用论文中提出的69个字符来表征输入数据
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
#  alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    sequenceLength = 1014  # 字符表示的序列长度
    batchSize = 128
    rate = 0.8  # 训练集的比例
    dataSource = "../data/preProcess/labeledCharTrain.csv"
    training = TrainingConfig()
    model = ModelConfig()
config = Config()