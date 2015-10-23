__author__ = 'heni'


def write_best_extraction(sentence_extractions, best_extractions_file):
    scores = []
    for element in sentence_extractions:
        scores.append(element.split('\t')[1])

    assert len(scores) == len(sentence_extractions)
    best_ext_ind = scores.index(max(scores))
    best_ext = sentence_extractions[best_ext_ind]

    best_extractions_file.write(best_ext)


def get_best_extractions_file(groundtruth_file_path):
    groundtruth_file = open(groundtruth_file_path, 'r')
    best_extractions_file = open(groundtruth_file_path.replace('.txt', '.oneExt.txt'), 'w')
    lines = groundtruth_file.readlines()

    i = 0
    sentence_extractions = []
    while i < len(lines):
        if i == len(lines) - 1:
            sentence_extractions.append(lines[i])
            write_best_extraction(sentence_extractions, best_extractions_file)
            i += 1
        elif lines[i].split('\t')[-1] == lines[i + 1].split('\t')[-1]:
            sentence_extractions.append(lines[i])
            i += 1
        else:
            sentence_extractions.append(lines[i])
            i += 1
            write_best_extraction(sentence_extractions, best_extractions_file)
            sentence_extractions = []

    groundtruth_file.close()
    best_extractions_file.close()

    return


def _get_spo(extractions):
    spo_extractions = extractions[0].split(';')
    assert len(spo_extractions) == 3
    spo_extraction_prep = []
    for phrase in spo_extractions:
        if not phrase[0].isalnum():
            phrase = phrase[1:]
        if not phrase[-1].isalnum():
            phrase = phrase[:-1]

        spo_extraction_prep.append(phrase)
    assert len(spo_extraction_prep) == 3
    spo_subsent = str(spo_extraction_prep[0] + ' ' + spo_extraction_prep[1] + ' ' + spo_extraction_prep[2])
    spo_words = spo_subsent.split(' ')
    return spo_words, spo_extraction_prep


def _get_words_groups(extractions,is_groundtruth):
    [spo_words, spo_extraction] = _get_spo(extractions)
    # context_words = []
    # context_label = ''
    context_dict={}
    if len(extractions) > 1:
        contexts=extractions[1:]
        for cont in contexts:
            if is_groundtruth:
                [context_label, context_words] = cont.split(':')
            else:
                [context_label, context_words] = cont.split('=')
            context_words=context_words.strip()
            not_prep_words=context_words.split(' ')
            prep_words=[]
            for word in not_prep_words:

                if not word[0].isalnum():
                    word=word[1:]
                try:
                    if not word[-1].isalnum():
                        word=word[:-1]
                except:
                    continue

                prep_words.append(word)

            context_dict[context_label]=prep_words

    return spo_words, spo_extraction, context_dict


def _add_spo_tags(elem, spo_extraction):
    if elem in spo_extraction[0]:
        if spo_extraction[0].index(elem) == 0:
            to_write = elem + '\t\t' + 'B-Subj' + '\n'
        else:
            to_write = elem + '\t\t' + 'I-Subj' + '\n'
    elif elem in spo_extraction[1]:
        if spo_extraction[1].index(elem) == 0:
            to_write = elem + '\t\t' + 'B-Pred' + '\n'
        else:
            to_write = elem + '\t\t' + 'I-Pred' + '\n'
    else:
        if spo_extraction[2].index(elem) == 0:
            to_write = elem + '\t\t' + 'B-Obj' + '\n'
        else:
            to_write = elem + '\t\t' + 'I-Obj' + '\n'
    return to_write


def _add_context_tags(word, cont_dict,is_groundtruth):
    if is_groundtruth:
        cont_labels_dictionary={'a':'attrib','e':'enabler','l':'location','t':'time'}
    else:
        cont_labels_dictionary={'attrib':'attrib','enabler':'enabler','l':'location','t':'time'}
    to_write=''
    cont_keys=list(cont_dict.keys())
    cont_values=list(cont_dict.values())
    assert len(cont_keys)==len(cont_values)
    i=0
    while i<len(cont_values):
        if word in cont_values[i]:
            ind=i
            break
        else:
            i+=1
    key=cont_keys[ind]
    values=cont_values[ind]
    new_label=cont_labels_dictionary[key]
    if values.index(word)==0:
        to_write = word + '\t\t' + 'B-'+new_label+ '\n'
    else:
        to_write = word + '\t\t' + 'I-'+new_label+ '\n'

    return to_write


def get_iob(sentence, extractions,is_groundtruth):
    to_write = ''
    [spo_words, spo_extraction, cont_dict] = _get_words_groups(extractions,is_groundtruth)
    cont_dict_values=list(cont_dict.values())
    assert len(cont_dict_values)==len(list(cont_dict.keys()))
    cont_words=[]
    for i in cont_dict_values:
        cont_words+=i
    sentence_elements = sentence.split(' ')
    for word in sentence_elements:
        if word in spo_words:
            to_write += _add_spo_tags(word, spo_extraction)
        elif word in cont_words:
            to_write += _add_context_tags(word, cont_dict,is_groundtruth=is_groundtruth)
        else:
            to_write += word + '\t\t' + 'O'+'\n'

    return to_write
