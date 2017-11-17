import tensorflow as tf
import collections as col
import os
import sys

#python版本
Py3 = sys.version_info[0] == 3


#返回单词列表,替换'\n'
def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        if Py3:
            return f.read().replace('\n', '<eso>').split()
        else:
            return f.read().decode('utf-8').replace('\n', '<eso>').split()


#每个单词对应唯一id
def _build_vocab(filename):
    data = _read_words(filename)
    counter = col.Counter(data)     #统计词频 (word:count)
    print(counter)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) #按照词频排序
    print(count_pairs)
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


# 文件里的单词变成对应的id
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# 返回训练,验证,测试集
def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


#把数据和标签分成若干个batch返回
def ptb_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size*batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])

        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i*num_steps+1], [batch_size, (i+1)*num_steps+1])

        y.set_shape([batch_size, num_steps])
        return x, y


if __name__ == '__main__':
    # word_to_id = _build_vocab('data_test')
    # print(word_to_id)
    print(sys.version_info)