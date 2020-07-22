# Author:yifan

import tensorflow as tf
import parameter_config
import get_train_data
config =parameter_config.Config()
data = get_train_data.Dataset(config)

#7预测代码
x = "this movie is full of references like mad max ii the wild one and many others the ladybug´s face it´s a clear reference or tribute to peter lorre this movie is a masterpiece we´ll talk much more about in the future"
# x = "This film is not good"   #最终反馈为1
# x = "This film is   bad"   #最终反馈为0
# x = "This film is   good"   #最终反馈为1

# 根据前面get_train_data获取，变成可以用来训练的向量。
y = list(x)
data._genVocabulary(y)
print(x)
reviewVec = data._reviewProcess(y, config.sequenceLength, data._charToIndex)
print(reviewVec)

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # 恢复模型
        checkpoint_file = tf.train.latest_checkpoint("../model/charCNN/model/")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX          = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # 获得输出的结果
        predictions = graph.get_tensor_by_name("outputLayer/binaryPreds:0")
        pred = sess.run(predictions, feed_dict={inputX: [reviewVec], dropoutKeepProb: 1.0,})[0]

# pred = [idx2label[item] for item in pred]
print(pred)