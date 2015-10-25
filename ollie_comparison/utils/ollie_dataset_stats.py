__author__ = 'heni'

import nltk


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

    return sentences_pos_tags


def get_context_stats(dataseth_path):
    dataset_file=open(dataseth_path,'r')
    dataset_lines=dataset_file.readlines()
    contex_dict={'zero':0,'one':0,'two':0,'three':0,'four':0}
    cont_ext_num=0
    for line in dataset_lines:
        cont_sep=line.split(')(')
        if line.count(')(')>0:
            cont_ext_num+=1

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

    return cont_ext_num,contex_dict


# [cont_num,cont_dict]=get_context_stats('data/ollie-scored.oneExt.txt')
# print(cont_num)
# print(list(cont_dict.keys()))
# print(list(cont_dict.values()))

