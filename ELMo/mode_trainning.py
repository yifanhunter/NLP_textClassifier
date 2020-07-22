# Author:yifan
# _*_ coding:utf-8 _*_
import os
import datetime
import numpy as np
import tensorflow as tf
import parameter_config
import get_train_data
import mode_structure
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from data import TokenBatcher   #不能从bilm直接导入TokenBatcher，因为需要修改内部的open为with open(filename, encoding="utf8") as f:
from bilm import  BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher

#获取前些模块的数据
config =parameter_config.Config()
data = get_train_data.Dataset(config)
data.dataGen()

#4生成batch数据集
def nextBatch(x, y, batchSize):
    # 生成batch数据集，用生成器的方式输出
    # perm = np.arange(len(x))  #返回[0  1  2  ... len(x)]的数组
    # np.random.shuffle(perm)  #乱序
    # # x = x[perm]
    # # y = y[perm]
    # x = np.array(x)[perm]
    # y = np.array(y)[perm]
    # print(x)
    # # np.random.shuffle(x)  #不能用这种，会导致x和y不一致
    # # np.random.shuffle(y)

    midVal = list(zip(x, y))
    np.random.shuffle(midVal)
    x, y = zip(*midVal)
    x = list(x)
    y = list(y)
    print(x)
    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end])
        batchY = np.array(y[start: end])
        yield batchX, batchY

# 5 定义计算metrics的函数
"""
定义各类性能指标
"""
def mean(item):
    return sum(item) / len(item)
def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)
    
    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)
	
# 6 训练模型
# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

# 定义计算图

with tf.Graph().as_default():

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率  

    sess = tf.Session(config=session_conf)
    
    # 定义会话
    with sess.as_default():
        elmoMode = mode_structure.ELMo(config)
        
        # 实例化BiLM对象，这个必须放置在全局下，不能在elmo函数中定义，否则会出现重复生成tensorflow节点。
        with tf.variable_scope("bilm", reuse=True):
            bilm = BidirectionalLanguageModel(
                    config.optionFile,
                    config.weightFile,
                    use_character_inputs=False,
                    embedding_weight_file=config.tokenEmbeddingFile
                    )
        inputData = tf.placeholder('int32', shape=(None, None))
        
        # 调用bilm中的__call__方法生成op对象
        inputEmbeddingsOp = bilm(inputData) 
        
        # 计算ELMo向量表示
        elmoInput = weight_layers('input', inputEmbeddingsOp, l2_coef=0.0)
        
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(elmoMode.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        
        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        
        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))
        
        lossSummary = tf.summary.scalar("loss", elmoMode.loss)
        summaryOp = tf.summary.merge_all()
        
        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
        
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
        
        
        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        savedModelPath ="../model/ELMo/savedModel"
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)

        # 保存模型的一种方式，保存为pb文件
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())
    
        def elmo(reviews):
            """
            对每一个输入的batch都动态的生成词向量表示
            """
#           tf.reset_default_graph()
            # TokenBatcher是生成词表示的batch类
            # print("________")
            batcher = TokenBatcher(config.vocabFile)
            # 生成batch数据
            inputDataIndex = batcher.batch_sentences(reviews)
            # 计算ELMo的向量表示
            elmoInputVec = sess.run(
                [elmoInput['weighted_op']],
                 feed_dict={inputData: inputDataIndex}
            )
            return elmoInputVec

        def trainStep(batchX, batchY):
            """
            训练函数
            """   
            
            feed_dict = {
              elmoMode.inputX: elmo(batchX)[0],  # inputX直接用动态生成的ELMo向量表示代入
              elmoMode.inputY: np.array(batchY, dtype="float32"),
              elmoMode.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, elmoMode.loss, elmoMode.predictions, elmoMode.binaryPreds],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss, acc, auc, precision, recall))
            trainSummaryWriter.add_summary(summary, step)

        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
              elmoMode.inputX: elmo(batchX)[0],
              elmoMode.inputY: np.array(batchY, dtype="float32"),
              elmoMode.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, elmoMode.loss, elmoMode.predictions, elmoMode.binaryPreds],
                feed_dict)
            
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            
            evalSummaryWriter.add_summary(summary, step)
            
            return loss, acc, auc, precision, recall
        
        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep) 
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")
                    
                    losses = []
                    accs = []
                    aucs = []
                    precisions = []
                    recalls = []
                    
                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, auc, precision, recall = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        aucs.append(auc)
                        precisions.append(precision)
                        recalls.append(recall)
                        
                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str, currentStep, mean(losses), 
                                                                                                       mean(accs), mean(aucs), mean(precisions),
                                                                                                       mean(recalls)))
                    
                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, "../model/ELMo/model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))
                    
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(elmoMode.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(elmoMode.dropoutKeepProb)}

        outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(elmoMode.binaryPreds)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

        builder.save()

