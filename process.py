__author__ = 'heni'

import os.path
import pickle
from preprocess.wordemb import WordEmbeddings,create_word_index
from preprocess.preprocess import get_iob


# this function aims to train a RNN
# it takes as input:
#   network_path: path to folder containing specific RNN configuration
#   articles: list of articles on which the network will make his training
def train_network(network_path,articles):
    if not os.path.isdir(network_path):
        raise Exception('Specified path %s is not a directory. \n Or a RNN need to be created' % network_path)
    settings = {'partial_training':0.95,
                    'partial_testing':0.05,
                    'fold':3, # 5 folds 0,1,2,3,4
                    'lr':0.0627142536696559,
                    'verbose':1,
                    'decay':False, # decay on the learning rate if improvement stops
                    'win':7, # number of words in the context window
                    'bs':9, # number of backprop through time steps
                    'nhidden':100, # number of hidden units
                    'seed':345,
                    'emb_dimension':100, # dimension of word embedding
                    'nepochs':50}

    # loading words' indices from the stored pickle file

    return settings


# This function allows to create a dictionary from an already generated .iob file
def get_dict_from_iob(article_name,dictionary):
    try:
        article = open('./data/iob/' + str(article_name) + '.iob','r')
    except:
        print('No .iob file found for: %s .' % article_name)

    print('The .iob file for %s has been found and a word2index dictionary will be created' % article_name)
    lines=article.readlines()
    i=0
    while i<len(lines):
        line=lines[i]
        if '\t\t' in line:
            w_l=line.split('\t\t')
            dictionary['wordIndex'].add_word(w_l[0])
            dictionary['labelIndex'].add_word(w_l[1].replace('\n',''))
            i+=1
        else:
            i+=1

    return dictionary


# main process
def run_process(articles):
    indices={
        'wordIndex':WordEmbeddings(),
        'labelIndex':WordEmbeddings()
    }
    indices_dict_path='./data/word2indices/indices.pickle'
    for i in range(0,len(articles)):
        article=articles[i]
        get_iob(article)
        if os.path.isfile(indices_dict_path):
            print('Loading indices dictionary from %s' % indices_dict_path)
            existent_indices_dict=pickle.load(open(indices_dict_path,'rb'))
            indices['wordIndex'].merge(existent_indices_dict['wordIndex'])
            indices['labelIndex'].merge(existent_indices_dict['labelIndex'])
        else:
            if i>0:
                raise Exception('INDICES DICTIONARY CREATED. NO NEED TO CREATE IT AGAIN !!!!!!')
            print('No indices dictionary found. This will be created based on the first article: %s' % str(article))
            indices['wordIndex'].merge(get_dict_from_iob(article,indices))
            indices['labelIndex'].merge(get_dict_from_iob(article,indices))
            print('The created dictionary is going to be stored.')
            with open(indices_dict_path,'wb') as indices_dict:
                pickle.dump(indices,indices_dict)
            print('The dictionary was successfully saved. Path: %s' % indices_dict_path)











