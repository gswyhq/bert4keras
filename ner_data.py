from __future__ import print_function
import sys
import numpy
from keras.preprocessing.sequence import pad_sequences

def load_data(train_path='data/ner_rel_train_BIOES.txt', test_path='data/ner_rel_test_BIOES.txt', maxlen=None, min_freq=2):

    train = _parse_data(train_path)
    test = _parse_data(test_path)

    # word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = ['<pad>', '<unk>']
    chunk_tags = set()
    vocab_set = set()

    # vocab += [w for w, f in iter(word_counts.items()) if f >= min_freq]
    for sample in train:
        for row in sample:
            if row[1] != '/':
                continue
            char = row[0]
            vocab_set.add(char)

    for sample in train + test:
        for row in sample:
            if row[1] != '/':
                continue
            tag = row[2:].upper()
            chunk_tags.add(tag)

    vocab = vocab + list(vocab_set)

    # in alphabetic order
    # pos_tags = sorted(list(set(row[1] for sample in train + test for row in sample)))
    # in alphabetic order
    # chunk_tags = sorted(list(set(row[2] for sample in train + test for row in sample)))
    chunk_tags = sorted(list(chunk_tags))
    train = _process_data(train, vocab,  chunk_tags, maxlen=maxlen)
    test = _process_data(test, vocab,  chunk_tags, maxlen=maxlen)
    return train, test, (vocab, chunk_tags)


def _parse_data(file_path):
    data = []
    with open(file_path)as f:
        for sample in f.readlines():
            if not sample or not sample.strip():
                continue
            row = sample.strip().rsplit('\t', maxsplit=1)[0]
            data.append(row.split())
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    # set to <unk> (index 1) if not in vocab
    x = [[word2idx.get(w[0].lower(), 1) for w in s if w[1] == '/'] for s in data]
    # for s in data:
    #     for w in s:
    #         if not w[2:]:
    #             print(s)
    #             sys.exit(0)
    y_chunk = [[chunk_tags.index(w[2:].upper()) for w in s if w[1] == '/'] for s in data]
    x = pad_sequences(x, maxlen)  # left padding

    # lef padded with -1. Indeed, any integer works as it will be masked
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk

def main():
    test_path = '/home/gswyhq/github_projects/bert4keras/data/ner_rel_dev_BIOES_5k.txt'
    data = _parse_data(test_path)
    print(data[0])

if __name__ == '__main__':
    main()
