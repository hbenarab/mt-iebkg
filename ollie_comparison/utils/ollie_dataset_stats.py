__author__ = 'heni'

import nltk
import collections


def _get_pos_index(pos,pos_ind_dict):
    ind=

def get_dataset_pos_tags(dataset_path):
    dataset_file=open(dataset_path,'r')
    dataset_lines=dataset_file.readlines()
    sentences_pos_tags=[]
    for i in range(0,len(dataset_lines)):
        sentence=dataset_lines[i][:-3]
        sentences_pos_tags.append([])
        tokens=nltk.word_tokenize(sentence)
        words_tags=nltk.pos_tag(tokens)
        assert len(words_tags)==len(tokens)
        for elem in words_tags:
            sentences_pos_tags[-1].append(elem[1])
        assert len(sentences_pos_tags[-1])==len(tokens)

    assert len(sentences_pos_tags)==len(dataset_lines)
    dataset_file.close()

    pos_ind_dict={}
    max_key=0
    for item in sentences_pos_tags:
        if item in list(pos_ind_dict.values()):
            continue
        else:
            try:
                pos_ind_dict[max(list(pos_ind_dict.keys()))+1]=item
            except:
                pos_ind_dict[1]=item

    pos2ind= _get_pos_index(pos,pos_ind_dict)
    # duplicated_tags=[item for item, count in collections.Counter(sentences_pos_tags).items() if count>1]

    return sentences_pos_tags


def get_context_stats(dataseth_path):
    dataset_file=open(dataseth_path,'r')
    dataset_lines=dataset_file.readlines()
    contex_dict={'zero':0,'one':0,'two':0,'three':0,'four':0}
    cont_ext_num=0
    # a_counter=0
    # e_counter=0
    # l_counter=0
    # t_counter=0
    counter_dict={'a':0,'e':0,'l':0,'t':0}
    for line in dataset_lines:
        cont_sep=line.split(')(')
        if ')(' in line:
            cont_ext_num+=1
        if line.count(')(a:')>0:
            counter_dict['a']+=1
        if line.count(')(e:')>0:
            counter_dict['e']+=1
        if line.count(')(l:')>0:
            counter_dict['l']+=1
        if line.count(')(t:')>0:
            counter_dict['t']+=1

        if len(cont_sep)==1:
            contex_dict['zero']+=1
        elif len(cont_sep)==2:
            contex_dict['one']+=1
        elif len(cont_sep)==3:
            contex_dict['two']+=1
        elif len(cont_sep)==4:
            contex_dict['three']+=1
        else:
            contex_dict['four']+=1

    assert cont_ext_num==(contex_dict['one']+contex_dict['two']+contex_dict['three']+contex_dict['four'])
    dataset_file.close()

    return cont_ext_num,contex_dict, counter_dict


[pos_tags,duplicates]=get_dataset_pos_tags('data/ollie-scored.oneExt.txt')


# [cont_num,cont_dict,counter_dict]=get_context_stats('data/ollie-scored.oneExt.txt')
# print(cont_num)
# print(list(cont_dict.keys()))
# print(list(cont_dict.values()))
# print(list(counter_dict.keys()))
# print(list(counter_dict.values()))

