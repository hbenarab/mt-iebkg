__author__ = 'heni'

import os


# This function aims to find the specific label of a phrase in the sentence
def find_label(element, labels):
    c = 0
    found = False
    while c < len(labels) and not found:
        if labels[c] in element:
            found = True
        else:
            c += 1
    return labels[c]


# This function convert the label from openIE output to english part-of-sentence label
def convert_label(openIE_label, elementInd, relationInd, myLabels):
    if openIE_label == 'SimpleArgument(':
        if elementInd < relationInd:
            label = myLabels[0]
        else:
            label = myLabels[2]
    elif openIE_label == 'Relation(':
        label = myLabels[1]
    elif openIE_label == 'Context(':
        label = myLabels[3]
    elif openIE_label == 'TemporalArgument(':
        label = myLabels[4]
    else:
        label = myLabels[5]
    return label


# This function allows to write a specific sentence's element in the IOB output file
# the element is written with its corresponding label in the IOB format
def write_element_to_iob_file(element, new_label, old_label):
    text_with_label = element.split(',List(')[0]
    text = text_with_label.replace(old_label, '')
    input_text_in_list = text.split(' ')
    output_text = []
    for i in range(0, len(input_text_in_list)):
        if i == 0:
            output_text.append(str(input_text_in_list[i]) + '\t\t' + 'B-' + str(new_label) + '\n')
        else:
            output_text.append(str(input_text_in_list[i]) + '\t\t' + 'I-' + str(new_label) + '\n')

    return output_text


# This function takes as input an openIE file of the article
# it returns an output file under the IOB format
def convert_openie2iob(article_name):
    article = open('../qa-jbt/data/openie' + str(article_name) + '.openie')
    lines = article.readlines()[1:]  # the first line containing the date of the file's creation is ignored
    iob_path = '../data/iob/' + str(article_name) + '.iob'
    if not os.path.isfile(iob_path):
        iob_file = open(iob_path, 'w')  # open new .iob file to write
        simpleArg_label = 'SimpleArgument('
        relation_label = 'Relation('
        context_label = 'Context('
        temporalArg_label = 'TemporalArgument('
        spatialArg_label = 'SpatialArgument('
        labels = [simpleArg_label, relation_label, context_label, temporalArg_label, spatialArg_label]
        subject = 'SUB'
        predicate = 'PRED'
        obj = 'OBJ'
        context = 'CONT'
        time = 'TIME'
        location = 'LOC'
        my_labels = [subject, predicate, obj, context, time, location]
        for i in range(0, len(lines)):
            line_elements = lines[i].split('\t')[2:]
            iob_file.write(line_elements[-1])
            try:
                relation_index = [i for i, s in enumerate(line_elements) if relation_label in s][0]
            except:
                print('Relation label not found in the sentence number %d' % i)

            for j in range(0, len(line_elements) - 2):
                element = line_elements[j]
                element_label = find_label(element, labels)
                new_label = convert_label(element_label, j, relation_index, my_labels)
                to_write = write_element_to_iob_file(element, new_label, element_label)
                for c in range(0, len(to_write)):
                    iob_file.write(to_write[c])

            iob_file.write('\n')

        article.close()
        iob_file.close()

    else:
        print('The IOB file for the article %s has been previously created. No need to do so again !!' % article_name)
    return
