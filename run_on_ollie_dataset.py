__author__ = 'heni'

import datetime
import math
import numpy
import pickle

from utils.tools import get_accuracy
from ollie_comparison.utils.training_tools import create_word2ind,create_network,get_labeled_data
from utils.tools import shuffle


def run_on_ollie_dataset(iob_ollie_dataset_path,use_cross_validation):
    settings = {'partial_training': 0.8,
                'partial_testing': 0.2,
                'fold': 10,  # 5 folds 0,1,2,3,4
                'lr': 0.05,
                'verbose': 1,
                'decay': False,  # decay on the learning rate if improvement stops
                'win': 7,  # number of words in the context window
                'bs': 9,  # number of backprop through time steps
                'nhidden': 100,  # number of hidden units
                'seed': 345,
                'emb_dimension': 100,  # dimension of word embedding
                'nepochs': 50}

    # iob_ollie_dataset_file=open(iob_ollie_dataset_path,'r')
    indices=create_word2ind(iob_ollie_dataset_path)
    words_index=indices['wordIndex']
    labels_index=indices['labelIndex']
    word2index = words_index.getCurrentIndex()
    index2word = words_index.getIndex2Word()
    label2index = labels_index.getCurrentIndex()
    index2label = labels_index.getIndex2Word()

    vocsize=len(word2index)
    nclasses=len(label2index)
    new_network_folder = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')
    rnn,model_folder=create_network(settings,nclasses,vocsize,new_network_folder)
    print('RNN model created and saved under %s' % model_folder)

    [labeled_data,labeled_data_size]=get_labeled_data(iob_ollie_dataset_path)
    print('Labeled data size for articles: ',labeled_data_size)
    sentences_list, labels_list = labeled_data.getData()
    while [] in sentences_list:
        print('Empty sentences were found. They will be removed')
        empty=sentences_list.index([])
        sentences_list.pop(empty)
        labels_list.pop(empty)
    assert len(sentences_list)==len(labels_list)
    number_labeled_sentences = len(sentences_list)

    print('The training phase of the RNN model on the Ollie dataset will begin now')
    rnn=rnn.load(model_folder)

    #########################################################
    # training with consideration to parameters in settings #
    #########################################################
    if not use_cross_validation:
        print('No cross-validation techniques will be used in this training process')
        shuffle([sentences_list, labels_list], settings['seed'])
        training_size = int(math.floor(settings['partial_training'] * number_labeled_sentences))
        testing_size = int(math.floor(settings['partial_testing'] * number_labeled_sentences))
        print('Training size: [0:{0}] = {0}'.format(training_size))
        train_sentences = sentences_list[0:training_size]
        train_labels = labels_list[0:training_size]
        print('Testing size: [{0}:{1}] = {2}'.format(training_size, training_size + testing_size, testing_size))
        test_sentences = sentences_list[training_size:training_size + testing_size]
        test_labels = labels_list[training_size:training_size + testing_size]
    else:
        print('Cross validation will be used')



    ####################
    # training process #
    ####################
    # number_train_sentences = len(train_sentences)
    # number_train_labels_toGuess = sum([len(x) for x in test_labels])
    # print('Starting training with {0} labeled sentences in total for {1} epochs.'.
    #       format(number_train_sentences, settings['nepochs']))

    best_accuracy = -numpy.inf
    current_learning_rate = settings['lr']
    best_epoch = 0

    f1_of_best_acc=0
    conf_mat_of_best_acc=None

    for e in range(0, settings['nepochs']):
        print('Epoch {0}'.format(e))
        print('----------------------------------------------')

        if use_cross_validation:
            ####################
            # validation phase #
            ####################
            print('Validation phase in process')
            shuffle([sentences_list, labels_list], settings['seed'])
            divide_in_folds=lambda lst,sz:[lst[i:i+sz] for i in range(0,len(lst),sz)]
            if len(sentences_list)%settings['fold']==0:
                size_of_fold=math.floor(len(sentences_list)/settings['fold'])
            else:
                size_of_fold=(math.floor(len(sentences_list)/settings['fold']))+1
            sentences_in_folds=divide_in_folds(sentences_list,size_of_fold)
            labels_in_folds=divide_in_folds(labels_list,size_of_fold)
            assert len(sentences_in_folds)==settings['fold']
            assert len(sentences_in_folds)==len(labels_in_folds)
            all_validation_accuracies=[]
            for j in range(0,len(sentences_in_folds)):
                ex_tr_sent=sentences_in_folds[:]
                ex_tr_labels=labels_in_folds[:]

                # val_sent=sentences_in_folds[j]
                # val_labels=labels_in_folds[j]
                # assert len(val_sent)==len(val_labels)

                val_sent=ex_tr_sent.pop(j)
                val_labels=ex_tr_labels.pop(j)
                assert len(val_sent)==len(val_labels)
                assert len(ex_tr_sent)==len(ex_tr_labels)

                tr_sent=[]
                tr_labels=[]
                for c in range(0,len(ex_tr_sent)):
                    tr_sent.extend(ex_tr_sent[c])
                    tr_labels.extend(ex_tr_labels[c])

                assert len(tr_sent)==len(tr_labels)

                train_dict={'sentences':tr_sent,'labels':tr_labels}
                validation_dict={'sentences':val_sent,'labels':val_labels}

                print('Training the fold number %i will begin now' % (j+1))
                [current_validation_accuracy,f1,conf_mat]=get_accuracy(rnn,train_dict,validation_dict,word2index,label2index,settings,
                                                         current_learning_rate,e,index2word,is_validation=True)

                all_validation_accuracies.append(current_validation_accuracy)
            assert len(all_validation_accuracies)==settings['fold']
            mean_validation=sum(all_validation_accuracies)/len(all_validation_accuracies)
            if mean_validation>best_accuracy:
                best_accuracy=mean_validation
                f1_of_best_acc=f1
                conf_mat_of_best_acc=conf_mat
                print('New best validation accuracy: %2.2f%%' % best_accuracy)
                # rnn.save(model_folder)
                print('A new RNN has been saved.')
            else:
                print('Validation phase did not come up with a better accuracy (only %2.2f%%).'
                      '. A new epoch will begin' % mean_validation)
                # rnn=rnn.load(model_folder)
                #continue
        ##################
        # Training phase #
        ##################
        else:
            shuffle([train_sentences, train_labels], settings['seed'])
            print('Training in progress')
            # rnn=rnn.load(model_folder)
            # print('RNN saved during the validation phase has been loaded')
            training_dict={'sentences':train_sentences,'labels':train_labels}
            testing_dict={'sentences':test_sentences,'labels':test_labels}
            [testing_accuracy,f1,conf_mat]=get_accuracy(rnn,training_dict,testing_dict,word2index,label2index,settings,
                                          current_learning_rate,e,index2word,is_validation=False)

            print('Accuracy during the testing phase (number of correct guessed labels) at %2.2f%%.' % testing_accuracy)

            # check if current epoch is the best
            if testing_accuracy> best_accuracy:
                best_accuracy = testing_accuracy
                best_epoch = e
                f1_of_best_acc=f1
                conf_mat_of_best_acc=conf_mat
                print('Better testing accuracy !!')

        if abs(best_epoch-e)>=5:
            current_learning_rate*=0.5

        if current_learning_rate<1e-5: break

    print('BEST RESULT: epoch ', best_epoch, 'with best accuracy: ', best_accuracy, '.',)
    # iob_ollie_dataset_file.close()
    pickle.dump([best_accuracy,f1_of_best_acc,conf_mat_of_best_acc],open('perf.pck','wb'))

# import sys
# sys.path.append('/home/heni/git/masterThesisKG/mt-iebkg')
run_on_ollie_dataset('data/ollie-scored.iob.txt',use_cross_validation=False)