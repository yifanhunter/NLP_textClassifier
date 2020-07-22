# Author:yifan
import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
warnings.filterwarnings("ignore")
import parameter_config
import get_train_data
import mode_structure

#获取前些模块的数据
config =parameter_config.Config()
data = get_train_data.Dataset(config)
data.dataGen()

#4生成batch数据集
def nextBatch(x, y, batchSize):
    # 生成batch数据集，用生成器的方式输出
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    # print("++++++++++++++")
    # print(x)
    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")
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
    precision = precision_score(trueY, binaryPredY, average='macro')
    recall = recall_score(trueY, binaryPredY, average='macro')
    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)

# 6 训练模型
    # 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels
charEmbedding = data.charEmbedding

    # 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        cnn = mode_structure.CharCNN(config, charEmbedding)
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.RMSPropOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
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
        lossSummary = tf.summary.scalar("trainLoss", cnn.loss)

        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        builder = tf.saved_model.builder.SavedModelBuilder("../model/charCNN/savedModel")
        sess.run(tf.global_variables_initializer())

        def trainStep(batchX, batchY):
            """
            训练函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: config.model.dropoutKeepProb,
                cnn.isTraining: True
            }
            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss,
                                                                                               acc, auc, precision,
                                                                                               recall))
            trainSummaryWriter.add_summary(summary, step)

        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0,
                cnn.isTraining: False
            }
            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
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
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str,
                                                                                                       currentStep,
                                                                                                       mean(losses),
                                                                                                       mean(accs),
                                                                                                       mean(aucs),
                                                                                                       mean(
                                                                                                           precisions),
                                                                                                       mean(
                                                                                                           recalls)))

                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, "../model/charCNN/model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))

        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

        outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(cnn.binaryPreds)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
                                                                                      outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()