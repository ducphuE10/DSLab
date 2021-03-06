import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random


class MLP:
    def __init__(self,vocab_size,hidden_size):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size # số node ở layer 1

    def build_graph(self):
        self._X = tf.placeholder(tf.float32, shape=[None, self._vocab_size])
        self._real_Y = tf.placeholder(tf.int32, shape=[None, ])

        weights_1 = tf.get_variable(
            name='weights_input_hidden',
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        biases_1 = tf.get_variable(
            name='biases_input_hidden',
            shape=(self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2018)
        )

        weights_2 = tf.get_variable(
            name='weights_hidden_output',
            shape=(self._hidden_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        biases_2 = tf.get_variable(
            name='biases_hidden_output',
            shape=(NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018)
        )

        hidden = tf.matmul(self._X,weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden,weights_2) + biases_2

        labels_one_hot = tf.one_hot(indices=self._real_Y, depth= NUM_CLASSES, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,logits= logits)
        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs,axis = 1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    def trainer(self,loss,learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

class DataReader:
    def __init__(self,data_path,batch_size,vocab_size):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        self._data = []
        self._label= []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index,value = int(token.split(':')[0]), float(token.split(':')[1])
                vector[index] = value
            self._data.append(vector)
            self._label.append(label)

        self._data = np.array(self._data)
        self._label = np.array(self._label)
        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        start = self._batch_id*self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1 #mỗi lần duyệt hết toàn bộ dữ liệu => _num_epoch tăng lên 1
            self._batch_id = 0
            random.seed(2018)
            indices = np.random.permutation(len(self._data))
            self._data, self._label = self._data[indices],self._label[indices]

        return self._data[start:end],self._label[start:end]


def load_dataset():
    train_data_reader = DataReader(data_path='../datasets/20news-bydate/20news-train-processed_tf_idf.txt',
                                   batch_size=50,vocab_size=vocab_size)
    test_data_reader = DataReader(data_path='../datasets/20news-bydate/20news-test-processed_tf_idf.txt',
                                  batch_size=50,vocab_size=vocab_size)
    return train_data_reader, test_data_reader


# save_parameter(name = variable.name,value = variable.eval(),epoch = train_data_reader._num_epoch)
def save_parameter(name,value,epoch):
    filename = name.replace(':','-colon-') + '-epoch-{}.text'.format(epoch)
    if len(value.shape) == 1: #vector
        string_form = ','.join([str(number) for number in value])
    else: #matrix
        string_form = '\n'.join([','.join(str(number) for number in value[row]) for row in range(value.shape[0])])

    with open('../saved-paras/' + filename,'w') as f:
        f.write(string_form)

def restore_parameters(name, epoch):
    filename = name.replace(':','-colon-') + '-epoch-{}.text'.format(epoch)
    with open('../saved-paras/' + filename) as f:
        lines = f.read().splitlines()
    if len(lines) == 1:
        value = [float(number) for number in lines[0].split(',')]
    else:
        value = [[float(number) for number in lines[row].split(',')] for row in range(len(lines))]
    return value

if __name__ == '__main__':
    NUM_CLASSES = 20

    with open('../datasets/20news-bydate/word_idfs.txt') as f: #caculate vocab_size
        vocab_size = len(f.read().splitlines())
    mlp = MLP(vocab_size = vocab_size,hidden_size=50)
    predict_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss = loss,learning_rate = 0.1)
    train_data_reader, test_data_reader = load_dataset()

    #training
    with tf.Session() as sess:
        step  = 0
        MAX_STEP = 3000 #chọn epoch 10 nên chỉ lấy max_step =3000
        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            plabels_eval, loss_eval, _ = sess.run([predict_labels, loss, train_op],
                                                  feed_dict={mlp._X: train_data,mlp._real_Y: train_labels} )
            step += 1
            print('step: {}, loss: {}'.format(step, loss_eval))

            #Lưu tham số mô hình
            train_variables = tf.trainable_variables()
            for variable in train_variables:
                save_parameter(name = variable.name,
                               value = variable.eval(),
                               epoch = train_data_reader._num_epoch)

    #đánh giá trên dữ liệu test
    with tf.Session() as sess:
        epoch = 10
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            #lấy các tham số mô hình có name = variable.name, epoch = epoch
            saved_value = restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value) #gán giá trị cho các tham số
            sess.run(assign_op)

        num_true_preds = 0 #số lượng dự đoán đúng
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabels_eval = sess.run(predict_labels,
                                         feed_dict={mlp._X: test_data,mlp._real_Y: test_labels})
            matches = np.equal(test_plabels_eval, test_labels)
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader._batch_id == 0: # duyệt hết epoch đầu tiên => _batch_id = 0 => dừng
                break

        print('Epoch:', epoch)
        print('Accuracy on test data:', num_true_preds / len(test_data_reader._data))

        # Epoch: 10
        # Accuracy on test data: 0.7877057886351566
        # -------
        #Epoch: 44
        #Accuracy on test data: 0.7538502389803505

