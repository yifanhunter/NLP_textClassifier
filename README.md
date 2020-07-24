# NLP_textClassifier
基于word2vec预训练词向量； textCNN 模型 ；charCNN 模型 ；Bi-LSTM模型；Bi-LSTM + Attention 模型 ；Transformer 模型 ；ELMo 预训练模型 ；BERT 预训练模型的文本分类项目


一、大纲概述
文本分类这个系列将会有8篇左右文章，从github直接下载代码，从百度云下载训练数据，在pycharm上导入即可使用，包括基于word2vec预训练的文本分类，与及基于近几年的预训练模型（ELMo，BERT等）的文本分类。总共有以下系列：
word2vec预训练词向量
textCNN 模型
charCNN 模型
Bi-LSTM 模型
Bi-LSTM + Attention 模型
Transformer 模型
ELMo 预训练模型
BERT 预训练模型

二、数据集合
数据集为IMDB 电影影评，总共有三个数据文件，在/data/rawData目录下，包括unlabeledTrainData.tsv，labeledTrainData.tsv，testData.tsv。在进行文本分类时需要有标签的数据（labeledTrainData），但是在训练word2vec词向量模型（无监督学习）时可以将无标签的数据一起用上。
训练数据地址：链接：https://pan.baidu.com/s/1-XEwx1ai8kkGsMagIFKX_g     提取码：rtz8

相关的介绍：https://www.cnblogs.com/yifanrensheng/category/1758378.html
