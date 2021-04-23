import numpy as np

def get_tf_idf(data_path):

    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        word_IDs = dict([(word,index) for index,(word,idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)

    with open(data_path) as f:
        documents = [(int(line.split('<fff>')[0]),
                      int(line.split('<fff>')[1]),
                      line.split('<fff>')[2]) for line in f.read().splitlines()]
    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        words_tdidfs = []
        sum_squares = 0.0
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = term_freq/max_term_freq *idfs[word]
            words_tdidfs.append((word_IDs[word],tf_idf_value))
            sum_squares += tf_idf_value**2

        words_tdidfs_normalized = [str(index) + ':' + str(tf_idf_value/np.sqrt(sum_squares))
                                   for index,tf_idf_value in words_tdidfs]

        sparse_rep = ' '.join(words_tdidfs_normalized)
        data_tf_idf.append((label,doc_id,sparse_rep))

    with open('../datasets/20news-bydate/data_tf-idf.txt','w') as f:
        for i in data_tf_idf:
            f.write(str(i[0]) + '<fff>'+str(i[1])+'<fff>'+i[2]+'\n')
