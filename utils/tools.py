__author__ = 'heni'

import random
import math
import time
import numpy
import sklearn.metrics
from preprocess.wordemb import WordEmbeddings


def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in range(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in range(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = math.floor(win/2) * [-1] + l + math.floor(win/2) * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out


def get_accuracy(rnn,train_set,test_set,word2index,label2index,settings,learning_rate,e,index2label,is_validation):
    train_sentences=train_set['sentences']
    train_labels=train_set['labels']
    assert len(train_sentences)==len(train_labels)
    test_sentences=test_set['sentences']
    test_labels=test_set['labels']
    assert len(test_sentences)==len(test_labels)

    number_train_labels_toGuess = sum([len(x) for x in test_labels])

    # word2index=indices_dicts['words']
    # label2index=indices_dicts['labels']

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
            rnn.train(word, label, learning_rate)
            rnn.normalize()
        # print(str(i)+' trained and normalized')
    if settings['verbose'] and not is_validation:
        # print('verbose')
        print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / len(train_sentences)),
              'completed in %.2f (sec) <<\r' % (time.time() - tic),flush=True)
    if settings['verbose'] and is_validation:
        print('[Validation] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / len(train_sentences)),
              'completed in %.2f (sec) <<\r' % (time.time() - tic),flush=True)

    # test_predictions=list(map(lambda x:index2label[x],
    #                   rnn.classify(numpy.asarray(contextwin(x,settings['win'])).astype('int32')))
    #                       for x in [[word2index[word] for word in sentence] for sentence in test_sentences])

    test_predictions=[]
    for sent in test_sentences:
        ind_sent=[word2index[w] for w in sent]
        prediction=rnn.classify(numpy.asarray(contextwin(ind_sent,settings['win'])).astype('int32'))
        #indexed_prediction=[index2label[p] for p in prediction]
        test_predictions.append(prediction)



    indexed_test_labels=[[label2index[w] for w in sent_labels] for sent_labels in test_labels]
    # correctGuesses_list = [[1 if pred_val == exp_val else 0 for pred_val, exp_val in zip(pred, exp)]
    #                        for pred, exp in zip(test_predictions, test_labels)]
    #
    # correct_guesses = (sum([sum(x) for x in correctGuesses_list]))
    # accuracy = correct_guesses * 100. / number_train_labels_toGuess
    assert len(test_predictions)==len(test_labels)
    assert len(test_predictions)==len(indexed_test_labels)

    flat_truth=[item for sublist in indexed_test_labels for item in sublist]
    flat_predictions=[item for sublist in test_predictions for item in sublist]
    assert len(flat_predictions)==len(flat_truth)

    accuracy=sklearn.metrics.accuracy_score(flat_truth,flat_predictions)*100
    return accuracy