__author__ = 'heni'

import os

# os.getcwd()
# f=open('../qa-jbt/data/openie/Paris.openie','r')
# lines=f.readlines()
# print(lines[0])
# print(lines[1])
# print(lines[2])


def convert_openie2iob(article_name):
    article=open(os.path.join('../qa-jbt/data/openie',article_name))
    lines=article.readlines()[1:] # the first line containing the date of the file's creation is ignored
    iob_file=open('../data/iob/'+article_name+'.iob','w')
    for i in range(0,len(lines)):
        line_elements=lines[i].split('\t')[2:]


    return

