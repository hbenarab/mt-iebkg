__author__ = 'heni'

import numpy
import os

from preprocess.wordemb import WordEmbeddings
from preprocess.labeledText import LabeledText
from rnn.elman_model import Elman


def get_data_from_iob(iob_dataset_file):
    lines = iob_dataset_file.readlines()
    data_to_add = [[], []]
    data_to_add[0].append([])
    data_to_add[1].append([])
    print('Getting labeled data from the Ollie groundtruth')
    for i in range(0, len(lines)):
        line = lines[i]
        if line == '\n':  # new sentence
            data_to_add[0].append([])
            data_to_add[1].append([])
            assert len(data_to_add[0]) == len(data_to_add[1])
            continue
        if '\t\t' not in line:
            continue
        w_l = line.split('\t\t')
        data_to_add[0][len(data_to_add[0]) - 1].append(w_l[0])
        data_to_add[1][len(data_to_add[1]) - 1].append(w_l[1].replace('\n', ''))
        assert len(data_to_add[0]) == len(data_to_add[1])

    assert len(data_to_add[0]) == len(data_to_add[1])
    return data_to_add


def get_labeled_data(iob_dataset_file):
    labeled_data=LabeledText()
    data_length=[]
    data_to_add = get_data_from_iob(iob_dataset_file)
    labeled_data.addData(data_to_add)
    data_length.append(len(labeled_data.getData()[0]))

    print('Labeled data from the Ollie groundtruth created')
    return labeled_data,data_length


def create_network(settings, classes_number, vocab_size, folder):
    print('Building RNN model...')
    numpy.random.seed(settings['seed'])
    rnn = Elman(nh=settings['nhidden'],
                nc=classes_number,
                ne=vocab_size,
                de=settings['emb_dimension'],
                cs=settings['win'])
    rnn_fold = os.path.join('./data/rnnElman', folder)
    os.makedirs(rnn_fold)
    rnn.save(rnn_fold)
    return rnn,rnn_fold


def get_dict_from_iob(iob_dataset_file, wordEmb):

    print('The .iob file for %s has been found and a word2index dictionary will be created')
    lines = iob_dataset_file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if '\t\t' in line:
            w_l = line.split('\t\t')
            wordEmb['wordIndex'].add_word(w_l[0])
            wordEmb['labelIndex'].add_word(w_l[1].replace('\n', ''))
            i += 1
        else:
            i += 1

    return wordEmb


def create_word2ind(iob_dataset_file):
    indices={
        'wordIndex': WordEmbeddings(),
        'labelIndex': WordEmbeddings()
    }
    print('Creating word embeddings for the Ollie groundtruth')
    indices['wordIndex'].merge(get_dict_from_iob(iob_dataset_file,indices)['wordIndex'])
    indices['wordIndex'].merge(get_dict_from_iob(iob_dataset_file,indices)['labelIndex'])
    print('Word embeddings dictionary created for the Ollie groundtruth')
    return indices
