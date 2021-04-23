import re
import os



def gather_20newsgroups_data():
    path = '../datasets/20news-bydate/'
    dirs = [path + dir_name + '/' for dir_name in os.listdir(path) if not os.path.isfile(path+dir_name)]

    train_dir,test_dir = dirs[1],dirs[0]
    list_newgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
    list_newgroups.sort()

    with open('../datasets/stop-words.txt') as f:
        stop_words = f.read().splitlines() #list of stop words

    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    def collect_data_from(parent_dir,newsgroup_list):
        data = []
        for group_id,newsgroup in enumerate(newsgroup_list):
            label = group_id

            dir_path = parent_dir + newsgroup +'/'

            files = [(filename,dir_path + filename) for filename in os.listdir(dir_path) if os.path.isfile(dir_path+filename)]
            files.sort()
            for filename,filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()

                    #remove stop words, stemming remain words
                    words = [stemmer.stem(word) for word in re.split('\W+',text) if word not in stop_words]
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data

    train_data = collect_data_from(
        parent_dir= train_dir,
        newsgroup_list=list_newgroups
    )

    test_data = collect_data_from(
        parent_dir= test_dir,
        newsgroup_list=list_newgroups
    )

    full_data = train_data + test_data
    with open('../datasets/20news-bydate/20news-train-processed.txt','w') as f:
        f.write('\n'.join(train_data))
    with open('../datasets/20news-bydate/20news-test-processed.txt', 'w') as f:
        f.write('\n'.join(test_data))
    with open('../datasets/20news-bydate/20news-full-processed.txt', 'w') as f:
        f.write('\n'.join(full_data))

