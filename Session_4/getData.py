import re
from collections import defaultdict
from os import listdir
from os.path import isfile

def get_data_and_vocab():
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + '/' + newsgroup + '/'

            files = [(filename, dir_path + filename) for filename in listdir(dir_path)
                     if isfile(dir_path + filename)]
            files.sort()
            label = group_id
            print('processing: {}-{}'.format(group_id, newsgroup))
            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                    words = re.split('\W+', text)
                    if word_count is not None: # nếu test là None, train != None
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data

    word_count = defaultdict(int)
    path = '../datasets/20news-bydate'
    # lấy đường dẫn thư mục train và thư mục test
    parts = []
    for dir_name in listdir(path):
        if not isfile(path + '/' + dir_name):
            parts.append(path + '/' + dir_name)
    train_path, test_path = (parts[0], parts[1]) if 'train' in parts[0] else (parts[1], parts[0])

    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]  # danh sách các thư mục trong thư mục train
    # ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',...]
    newsgroup_list.sort()

    train_data = collect_data_from(parent_path=train_path,
                                   newsgroup_list=newsgroup_list,
                                   word_count=word_count)
    vocab = [word for word, freq in zip(word_count.keys(), word_count.values()) if freq > 10]
    vocab.sort()
    with open('../datasets/w2v/vocab-raw.txt', 'w') as f:
        f.write('\n'.join(vocab))

    test_data = collect_data_from(parent_path=train_path, newsgroup_list=newsgroup_list)

    with open('../datasets/w2v/20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))

    with open('../datasets/w2v/20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))

get_data_and_vocab()