def encode_data(data_path,vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word,word_ID+2) for word_ID,word in enumerate(f.read().splitlines())])
        # list(vocab.items())[0:5]  ---> [('', 2), ('0', 3), ('00', 4), ('000', 5), ('0001', 6)]

    with open(data_path) as f:
        documents = [(line.split('<fff>')[0],(line.split('<fff>'))[1],line.split('<fff>'))[2]
                     for line in f.read().splitlines()]

    encoded_data = []
    MAX_DOC_LENGTH = 500
    unknown_ID = 0
    padding_ID = 1
    for document in documents:
        label, doc_id, text = document               #0, 49960, from mathew mathew mantis co...
        words = text.split()[:MAX_DOC_LENGTH]        #[from, mathew, mathew, mantis,..]
        sentence_length = len(words)                 # =500 nếu len(text)>=500, =len(text) nếu len(text)<500

        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_ID))
        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))


        encoded_data.append(str(label)+'<fff>' + str(doc_id)+'<fff>'+ str(sentence_length)
                            +'<fff>'+' '.join(encoded_text))

    dir_name = '/'.join(data_path.split('/')[:-1]) #../datasets/w2v
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt' #20news-train-encoded.txt

    with open(dir_name+'/'+file_name,'w') as f: #../datasets/w2v/20news-train-encoded.txt
        f.write('\n'.join(encoded_data))


test_path = '../datasets/w2vtest/20news-test-raw.txt'
train_path  = '../datasets/w2v/20news-train-raw.txt'
vocab_path = '../datasets/w2vtest/vocab-raw.txt'
encode_data(train_path,vocab_path)
encode_data(test_path,vocab_path)