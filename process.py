__author__ = 'heni'

import os.path
import pickle
import numpy
import datetime
import math
import time

from preprocess.wordemb import WordEmbeddings, create_word_index
from preprocess.preprocess import get_iob
from rnn.elman_model import Elman
from preprocess.labeledText import LabeledText
from utils.tools import shuffle, minibatch, contextwin


# this function aims to train a RNN
# it takes as input:
#   network_path: path to folder containing specific RNN configuration
#   articles: list of articles on which the network will make his training
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


def get_data_from_iob(article_name):
    try:
        article = open('./data/iob/' + str(article_name) + '.iob', 'r')
    except:
        raise Exception('No .iob file found for: %s .' % article_name)
    # labeled_data=[[],[]]
    lines = article.readlines()
    # beg=0
    # end=lines.index('\n')
    data_to_add = [[], []]
    data_to_add[0].append([])
    data_to_add[1].append([])
    print('Getting labeled data from the Wikipedia article: %s' % article_name)
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


# This function allows to create a dictionary from an already generated .iob file
def get_dict_from_iob(article_name, wordEmb):
    try:
        article = open('./data/iob/' + str(article_name) + '.iob', 'r')
    except:
        print('No .iob file found for: %s .' % article_name)

    print('The .iob file for %s has been found and a word2index dictionary will be created' % article_name)
    lines = article.readlines()
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


def create_word2ind(articles):
    indices={
        'wordIndex': WordEmbeddings(),
        'labelIndex': WordEmbeddings()
    }
    for i in range(0,len(articles)):
        article=articles[i]
        get_iob(article)
        print('Creating word embeddings for the article %s' %article)
        indices['wordIndex'].merge(get_dict_from_iob(article,indices)['wordIndex'])
        indices['wordIndex'].merge(get_dict_from_iob(article,indices)['labelIndex'])
    print('Word embeddings dictionary created for input the %i input article' % len(articles))
    return indices


def get_labeled_data(articles):
    labeled_data=LabeledText()
    for i in range(0,len(articles)):
        article=articles[i]
        data_to_add = get_data_from_iob(article)
        labeled_data.addData(data_to_add)
    print('Labeled data for the %i articles is created' % len(articles))
    return labeled_data


# main process: taking as input a list of articles to be processed
def run_process(articles):
    settings = {'partial_training': 0.7,
                'partial_testing': 0.3,
                'fold': 3,  # 5 folds 0,1,2,3,4
                'lr': 0.0627142536696559,
                'verbose': 1,
                'decay': False,  # decay on the learning rate if improvement stops
                'win': 7,  # number of words in the context window
                'bs': 9,  # number of backprop through time steps
                'nhidden': 100,  # number of hidden units
                'seed': 345,
                'emb_dimension': 100,  # dimension of word embedding
                'nepochs': 10}

    indices=create_word2ind(articles)
    word_index = indices['wordIndex']
    label_index = indices['labelIndex']
    word2index = word_index.getCurrentIndex()
    index2word = word_index.getIndex2Word()
    label2index = label_index.getCurrentIndex()
    index2label = label_index.getIndex2Word()
    vocsize = len(word2index)
    nclasses = len(label2index)

    new_network_folder = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')
    rnn,model_folder=create_network(settings,nclasses,vocsize,new_network_folder)
    print('RNN model created and saved under %s' % model_folder)

    labeled_data=get_labeled_data(articles)
    sentences_list, labels_list = labeled_data.getData()
    number_labeled_sentences = len(sentences_list)

    for i in range(0, len(articles)):
        article = articles[i]
        print('Training for article %s will begin now' % article)
        rnn=rnn.load(model_folder)
        shuffle([sentences_list, labels_list], settings['seed'])

        training_size = int(math.floor(settings['partial_training'] * number_labeled_sentences))
        testing_size = int(math.floor(settings['partial_testing'] * number_labeled_sentences))
        print('Training size: [0:{0}] = {0}'.format(training_size))
        train_sentences = sentences_list[0:training_size]
        train_labels = labels_list[0:training_size]
        print('Testing size: [{0}:{1}] = {2}'.format(training_size, training_size + testing_size, testing_size))
        test_sentences = sentences_list[training_size:training_size + testing_size]
        test_labels = labels_list[training_size:training_size + testing_size]

        ####################
        # training process #
        ####################
        number_train_sentences = len(train_sentences)
        number_train_labels_toGuess = sum([len(x) for x in test_labels])
        print('Starting training with {0} labeled sentences in total for {1} epochs.'.
              format(number_train_sentences, settings['nepochs']))
        best_accuracy = -numpy.inf
        current_learning_rate = settings['lr']
        best_epoch = 0
        for e in range(0, settings['nepochs']):
            print('Epoch {0}'.format(e))
            print('----------------------------------------------')
            shuffle([train_sentences, train_labels], settings['seed'])
            tic = time.time()
            for i in range(0, len(train_sentences)):
                # print(i)
                indexed_sentence = [word2index[w] for w in train_sentences[i]]
                indexed_labels = [label2index[l] for l in train_labels[i]]
                cs_window = contextwin(indexed_sentence, settings['win'])
                words = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(cs_window, settings['bs']))
                # words=[]
                # mini_batch=minibatch(cs_window,settings['bs'])
                # to_array=lambda x:numpy.asarray(x).astype('int32')
                # for i in range(0,len(mini_batch)):
                #     words.append(to_array(mini_batch[i]))
                # assert len(words)==len(indexed_labels)
                for word, label in zip(words, indexed_labels):
                    rnn.train(word, label, current_learning_rate)
                    rnn.normalize()
                # print(str(i)+' trained and normalized')
                if settings['verbose'] and i==len(train_sentences)-1:
                    # print('verbose')
                    print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / number_train_sentences),
                          'completed in %.2f (sec) <<\r' % (time.time() - tic),flush=True)

            # evaluation // back into the real world : idx -> words

            predictions_test=[map(lambda x:index2label[x],
                                  rnn.classify(numpy.asarray(contextwin(x,settings['win'])).astype('int32')))
                              for x in [[word2index[word] for word in sentence] for sentence in test_sentences]]
            # all_sentences_indices=[]
            # for sentence in test_sentences:
            #     sentence_indices=[]
            #     for word in sentence:
            #         sentence_indices.append(word2index[word])
            #     all_sentences_indices.append(sentence_indices)
            # predictions_test=[]
            # for x in all_sentences_indices:
            #

            correctGuesses_list = [[1 if pred_val == exp_val else 0 for pred_val, exp_val in zip(pred, exp)]
                                   for pred, exp in zip(predictions_test, test_labels)]

            correct_guesses = (sum([sum(x) for x in correctGuesses_list]))
            accuracy = correct_guesses * 100. / number_train_labels_toGuess
            print('Accuracy (number of correct guessed labels) at %2.2f%%.' % accuracy)

            # check if current epoch is the best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = e
                rnn.save(model_folder)
                print('Better accuracy ====> New RNN saved')
            else:
                current_learning_rate*=0.5
            if current_learning_rate<1e-5: break

        print('BEST RESULT: epoch ', best_epoch, 'with best accuracy: ', best_accuracy, '.')
        # rnn.save(model_folder)


# run_process(['Obama','Jupiter','Paris'])
run_process(['Jupiter','Paris'])