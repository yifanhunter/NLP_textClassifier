# Author:yifan
import logging
import gensim
from gensim.models import  word2vec
# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("../data/preProcess/wordEmbdiing.txt")
# a = list(sentences)
# print(len(a))

# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)
model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True)

# 加载bin格式的模型
word2Vec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin",binary=True)