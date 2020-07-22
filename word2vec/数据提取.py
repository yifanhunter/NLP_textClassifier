# 数据集为IMDB 电影影评，总共有三个数据文件，在/data/rawData目录下，
# 包括unlabeledTrainData.tsv，labeledTrainData.tsv，testData.tsv。
# 在进行文本分类时需要有标签的数据（labeledTrainData），但是在训练word2vec词向量模型（无监督学习）时可以将无标签的数据一起用上。
import pandas as pd
from bs4 import BeautifulSoup

# IMDB 电影影评属于英文文本，本序列主要是文本分类的模型介绍，因此数据预处理比较简单，
# 只去除了各种标点符号，HTML标签，小写化等。代码如下：

with open("../data/rawData/unlabeledTrainData.tsv", "r", encoding='UTF-8') as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

with open("../data/rawData/labeledTrainData.tsv", "r", encoding='UTF-8') as f:
    labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]

unlabel = pd.DataFrame(unlabeledTrain[1:], columns=unlabeledTrain[0])
label = pd.DataFrame(labeledTrain[1:], columns=labeledTrain[0])   #多一列数据 sentiment （0/1）
# print("```````````")
# print(unlabeledTrain)
# print("========")
# print(unlabel)
# print("________")
# print(label)

def cleanReview(subject):
 # 数据处理函数
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',',
                                                                                                          '').replace(
        '.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)
    return newSubject

unlabel["review"] = unlabel["review"].apply(cleanReview)
label["review"] = label["review"].apply(cleanReview)

# 将有标签的数据和无标签的数据合并
newDf = pd.concat([unlabel["review"], label["review"]], axis=0)
# 保存成txt文件
newDf.to_csv("../data/preProcess/wordEmbdiing.txt", index=False)