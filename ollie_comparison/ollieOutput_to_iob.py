__author__ = 'heni'

import os
import ollie_comparison.utils.preprocess_tools


def _ollie_output_to_log(ollie_groundtruth_file,log_file_name):
    # ollie_groundtruth=open(ollie_groundtruth_file,'r')
    # ollie_sentences=open('data/ollie_trainset.txt','r')

    # groundtruth_lines=ollie_groundtruth.readlines()
    # written_sentences=[]
    # for line in groundtruth_lines:
    #     parts=line.split('\t')
    #     sentence=parts[3]
    #     if sentence not in written_sentences:
    #         ollie_sentences.write(sentence)
    #         written_sentences.append(sentence)

    # ollie_sentences.close()
    # ollie_groundtruth.close()

    os.system("cd ollie/ && java -Xmx512m -jar ollie-app-latest.jar "
              "../data/ollie_trainset.txt >"
              " ../"+log_file_name)


def _spo_to_iob(spo_bloc):
    to_write=''
    assert len(spo_bloc)==3
    for i in range(0,len(spo_bloc)):
        label=spo_bloc[i].split(' ')
        if i==0:
            for elem in label:
                if not elem.isalnum():
                    to_write+=elem+'\t\t'+'O'+'\n'
                else:
                    first_subj_written=False
                    if label.index(elem)==0 and not first_subj_written:
                        to_write+=elem+'\t\t'+'B-Subj'+'\n'
                        first_subj_written=True
                    else:
                        to_write+=elem+'\t\t'+'I-Subj'+'\n'

        elif i==1:
            for elem in label:
                if not elem.isalnum():
                    to_write+=elem+'\t\t'+'O'+'\n'
                else:
                    first_pred_written=False
                    if label.index(elem)==0 and not first_pred_written:
                        to_write+=elem+'\t\t'+'B-Pred'+'\n'
                        first_pred_written=True
                    else:
                        to_write+=elem+'\t\t'+'I-Pred'+'\n'

        else:
            for elem in label:
                if not elem.isalnum():
                    to_write+=elem+'\t\t'+'O'+'\n'
                else:
                    first_obj_written=False
                    if label.index(elem)==0 and not first_obj_written:
                        to_write+=elem+'\t\t'+'B-Obj'+'\n'
                        first_obj_written=True
                    else:
                        to_write+=elem+'\t\t'+'I-Obj'+'\n'

    return to_write


def _context_to_iob(context_extraction):
    to_write=''
    context_sep=context_extraction[:-2].split('=')
    i=0
    context_phrase=context_sep[1].split(' ')
    for i in range(0,len(context_phrase)):
        elem=context_phrase[i]
        if not elem.isalnum():
            to_write+=elem+'\t\t'+'O'+'\n'
        else:
            if i==0:
                # to_write+='<B-'+str(context_sep[0])+'>'+elem+'</B-'+str(context_sep[0])+'>'+' '
                to_write+=elem+'\t\t'+'B-'+str(context_sep[0])+'\n'
            else:
                # to_write+='<I-'+str(context_sep[0])+'>'+elem+'</I-'+str(context_sep[0])+'>'+' '
                to_write+=elem+'\t\t'+'I-'+str(context_sep[0])+'\n'

    return to_write


def _write_to_output(ollie_output_file_iob,sentence_bloc):
    # ollie_output_file_iob.write(sentence_bloc[0])
    sentence_words=sentence_bloc[0][:-1].split(' ')
    if sentence_bloc[1]=='No extractions found.\n':
        # ollie_output_file_iob.write('No extractions found.\n')
        to_write=''
        for word in sentence_words:
            to_write+=word+'\t\t'+'O'+'\n'
        ollie_output_file_iob.write(sentence_bloc[0]+to_write)

    else:
        extraction=sentence_bloc[1].split(': (')[1:]
        assert len(extraction)==1
        spo_cont_separation=extraction[0].split(')[')
        spo_extraction=spo_cont_separation[0].split(';')
        assert len(spo_extraction)==3
        to_write=ollie_comparison.utils.preprocess_tools.get_iob(sentence_bloc[0][:-1],spo_cont_separation,is_groundtruth=False)
        ollie_output_file_iob.write(sentence_bloc[0]+to_write)

    ollie_output_file_iob.write('\n')


def ollie_output_to_iob(ollie_groundtruth_file,ollie_log_file,ollie_output_iob):
    _ollie_output_to_log(ollie_groundtruth_file,ollie_log_file)
    ollie_output_file=open(ollie_log_file,'r')
    ollie_output_lines=ollie_output_file.readlines()

    ollie_output_file_iob=open(ollie_output_iob,'w')

    ext_lines=ollie_output_lines[:]
    # sentence_bloc=[]
    i=0
    while i in range(0,len(ext_lines)):
        if ext_lines[i]=='\n':
            sentence_bloc=ext_lines[:i]
            _write_to_output(ollie_output_file_iob,sentence_bloc)
            ext_lines=ext_lines[i+1:]
            i=0
        else:
            i+=1

    ollie_output_file.close()
    ollie_output_file_iob.close()


ollie_output_to_iob('data/ollie-scored.txt','data/ollie_log.txt','data/ollie_output.iob.txt')