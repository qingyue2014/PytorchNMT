import sys, pickle, os, random
import numpy as np


def read_corpus(data_path, label_path, mode):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(data_path, encoding='utf-8') as fr1:
        sen_lines = fr1.readlines()
    with open(label_path, encoding='utf-8') as fr2:
        tag_lines = fr2.readlines()
    for index in range(0, len(sen_lines)):
        sen_line = sen_lines[index].replace("\n","").strip()
        tag_line = tag_lines[index].replace("\n","").strip()
        sent_ = sen_line.split(" ")
        tag_ = tag_line.split(" ")
        if (index+1) % 100 == 0:
            print("process {}_data {}/{}...".format(mode, (index+1),len(sen_lines)))
        data.append((sent_, tag_))
    if mode == 'train':
        with open('train_data.pkl', 'wb') as fw:
            pickle.dump(data, fw)
    else:
        with open('test_data.pkl', 'wb') as fw:
            pickle.dump(data, fw)
    return data


def vocab_build(vocab_path, tag_path, id2tag_path, data, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    word2id = {'<SOS>': 0, '<EOS>': 1,'<UNK>': 2}
    tag2label = {'<SOS>': 0, '<EOS>': 1, '<UNK>': 2}
    label2tag = {0: '<SOS>', 1: '<EOS>', 2: '<UNK>'}
    for sent_, tag_ in data:
        for word in sent_:
            if word not in word2id:
                word2id[word] = len(word2id)
        for tag in tag_:
            if tag not in tag2label:
                label2tag[len(tag2label)] = tag
                tag2label[tag] = len(tag2label)

    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)
    with open(tag_path, 'wb') as fw:
        pickle.dump(tag2label,fw)
    with open(id2tag_path, 'wb') as fw:
        pickle.dump(label2tag, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    sentence_id.append(word2id['<EOS>'])
    return sentence_id


def read_dictionary(vocab_path, tag_path, id2tag_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    with open(tag_path,'rb') as fr:
        tag2id = pickle.load(fr)
    with open(id2tag_path,'rb') as fr:
        id2tag = pickle.load(fr)
    return word2id, tag2id, id2tag