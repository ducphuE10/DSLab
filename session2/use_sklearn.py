import numpy as np

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]  # r_d = [0,0...,0] ( 14233 số 0 )
        indices_tfidfs = sparse_r_d.split()  # [13019:0.019366049587254193,13936:0.06773096056271098]
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()
        # [0<fff>49960<fff>13019:0.019366049587254193 13936:0.06773096056271098, .. ]

    with open('../datasets/20news-bydate/word_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())  # 14232

    data = []
    labels = []

    for data_id in d_lines:
        features = data_id.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])  # label = 0, doc_id = 49960
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        # features[2] = 13019:0.019366049587254193 13936:0.06773096056271098
        data.append(r_d)
        labels.append(label)

    return np.array(data),np.array(labels)

def compute_accuracy(predict_y,expected_y):
    matches = np.equal(predict_y,expected_y)
    accuracy = np.sum(matches.astype(float))/expected_y.size
    return accuracy

def clustering_with_Kmeans():
    data,labels = load_data(data_path = '../datasets/20news-bydate/data_tf_idf.txt')
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    from sklearn.metrics.cluster import completeness_score
    X = csr_matrix(data)
    print("=======")
    kmeans = KMeans(n_clusters=20,init='random',n_init=5,tol=1e-3,random_state=2018).fit(X)
    labels_predict = kmeans.labels_
    print("accuracy: ",completeness_score(labels, labels_predict))

def classifying_with_linear_SVM():
    train_X, train_y = load_data(data_path='../datasets/20news-bydate/20news-train-processed_tf_idf.txt')
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(C=10,tol = 0.001, verbose=True)
    classifier.fit(train_X,train_y)

    test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-processed_tf_idf.txt')
    predict_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predict_y = predict_y, expected_y = test_y)
    print('Accuracy:',accuracy)

def classifying_with_kernel_SVM():
    train_X, train_y = load_data(data_path='../datasets/20news-bydate/20news-train-processed_tf_idf.txt')
    from sklearn.svm import SVC
    classifier = SVC(C=50,kernel ='rbf',gamma=0.1,tol = 0.001, verbose=True)
    classifier.fit(train_X,train_y)

    test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-processed_tf_idf.txt')
    predict_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predict_y = predict_y, expected_y = test_y)
    print('Accuracy:',accuracy)


if __name__ == '__main__':
    classifying_with_linear_SVM()
    # classifying_with_kernel_SVM()
    # clustering_with_Kmeans()