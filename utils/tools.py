__author__ = 'heni'

import random
import math
import time
import numpy
import sklearn.metrics
import nltk


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


def _get_pos_tags(sentence):
    pos_tags = []
    # tokens=nltk.word_tokenize(sentence)
    words_tags=nltk.pos_tag(sentence)
    for doublet in words_tags:
        pos_tags.append(doublet[1])

    assert len(pos_tags)==len(sentence)
    return pos_tags


def _get_pos_indices(pos_tags,pos2ind_dict):
    pos_indices=[]

    for tag in pos_tags:
        pos_indices.append(pos2ind_dict[tag])

    return pos_indices


def _update_pos2ind_dict(pos_tags,pos2ind_dict):

    # keys=list(pos2ind_dict.keys())
    # values=list(pos2ind_dict.values())

    if not list(pos2ind_dict.values()):
        max_value=0
    else:
        max_value=max(list(pos2ind_dict.values()))

    for tag in pos_tags:
        if tag in list(pos2ind_dict.keys()):
            continue
        else:
            pos2ind_dict[tag]=max_value+1
            max_value+=1

    assert max(list(pos2ind_dict.values()))==max_value


def _get_cs_pos_tags(word,pos2ind_dict,index2word):
    cs_tags=[]
    for elem in word[0]:
        if elem==-1:
            cs_tags.append(-1)
        else:
            w=index2word[elem]
            pos_tag=nltk.pos_tag([w])[0][1]
            cs_tags.append(pos2ind_dict[pos_tag])

    assert len(word[0])==len(cs_tags)

    return cs_tags


def get_accuracy(rnn,train_set,test_set,word2index,label2index,settings,learning_rate,e,index2word,is_validation):
    train_sentences=train_set['sentences']
    train_labels=train_set['labels']
    assert len(train_sentences)==len(train_labels)
    test_sentences=test_set['sentences']
    test_labels=test_set['labels']
    assert len(test_sentences)==len(test_labels)

    number_train_labels_toGuess = sum([len(x) for x in test_labels])

    pos2ind_dict={}
    tic = time.time()
    for i in range(0, len(train_sentences)):
        # print(i)
        sentence=train_sentences[i]
        indexed_sentence = [word2index[w] for w in sentence]
        indexed_labels = [label2index[l] for l in train_labels[i]]
        cs_window = contextwin(indexed_sentence, settings['win'])
        words = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(cs_window, settings['bs']))
        pos_tags=_get_pos_tags(sentence)
        _update_pos2ind_dict(pos_tags,pos2ind_dict)
        pos_tags_indices=_get_pos_indices(pos_tags,pos2ind_dict)
        for word, pos, label in zip(words, pos_tags_indices, indexed_labels):
            # cs_pos_tags=_get_cs_pos_tags(word,pos2ind_dict,index2word)
            rnn.train(word, pos, label, learning_rate)
            rnn.normalize()
    if settings['verbose'] and not is_validation:
        print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / len(train_sentences)),
              'completed in %.2f (sec) <<\r' % (time.time() - tic),flush=True)
    if settings['verbose'] and is_validation:
        print('[Validation] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / len(train_sentences)),
              'completed in %.2f (sec) <<\r' % (time.time() - tic),flush=True)
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


