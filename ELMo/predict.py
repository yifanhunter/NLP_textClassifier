# Author:yifan
#测试没泡通
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import parameter_config
config =parameter_config.Config()
from data import TokenBatcher   #不能从bilm直接导入TokenBatcher，因为需要修改内部的open为with open(filename, encoding="utf8") as f:
from bilm import  BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher

#7预测代码
reviews = "this movie is full of references like mad max ii the wild one and many others the ladybug´s face it´s a clear reference or tribute to peter lorre this movie is a masterpiece we´ll talk much more about in the future"
# x = "his movie is the same as the third level movie. There's no place to look good"
# x = "This film is not good"   #最终反馈为0
# x = "This film is   bad"   #最终反馈为0

# 注：下面两个词典要保证和当前加载的模型对应的词典是一致的

x1 = [review[:200] for review in reviews]

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # 恢复模型
        checkpoint_file = tf.train.latest_checkpoint("../model/ELMo/model/")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

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
        def elmo(reviews):
            """
            对每一个输入的batch都动态的生成词向量表示
            """
            batcher = TokenBatcher(config.vocabFile)
            # 生成batch数据
            inputDataIndex = batcher.batch_sentences(reviews)
            # 计算ELMo的向量表示x
            elmoInputVec = sess.run(
                [elmoInput['weighted_op']],
                feed_dict={inputData: inputDataIndex}
            )
            return elmoInputVec


        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # 获得输出的结果
        binaryPreds = graph.get_tensor_by_name("output/binaryPreds:0") # mode_structure中的定义
        pred = sess.run(binaryPreds, feed_dict={inputX:elmo(x1)[0], dropoutKeepProb: 1.0})[0]

# print(pred)
# pred = [idx2label[item] for item in pred]
print(pred)