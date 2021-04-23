import numpy as np
from collections import defaultdict

def generate_vocabulary(data_path):
    def compute_idf(df,corpus_size):
        assert df >0
        return np.log10(corpus_size*1/df)
    with open(data_path) as f:
        lines = f.read().splitlines()

    doc_count = defaultdict(int)
    corpus_size = len(lines)
    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1

    words_idfs = [(word, compute_idf(document_freq, corpus_size))
                  for word, document_freq in zip(doc_count.keys(),doc_count.values())
                  if document_freq > 10 and not word.isdigit()]
    words_idfs.sort(key = lambda x: -x[1])
    print('Vocabulary size: {}'.format(len(words_idfs)))
    with open('../datasets/20news-bydate/words_idfs.txt','w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word,idf in words_idfs]))




