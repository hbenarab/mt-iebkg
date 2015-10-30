__author__ = 'heni'

import sklearn.metrics
import numpy

def get_ollie_iob_performance(ollie_output_file_path,ollie_groundtruth_file_path):
    ollie_output_file=open(ollie_output_file_path,'r')
    print('Ollie output file loaded: "%s"' % ollie_output_file_path)
    ollie_groundtruth_file=open(ollie_groundtruth_file_path,'r')
    print('Ollie groundtruth file loaded: "%s"' % ollie_groundtruth_file_path)
    output_lines=ollie_output_file.readlines()
    groundtruth_lines=ollie_groundtruth_file.readlines()
    assert len(output_lines)==len(groundtruth_lines)

    ground_labels=[[]]
    output_labels=[[]]
    for i in range(0,len(output_lines)):
        ground_line=groundtruth_lines[i]
        output_line=output_lines[i]
        if (ground_line=='\n') and (output_line=='\n'):
            ground_labels.append([])
            output_labels.append([])
            assert len(ground_labels)==len(output_labels)
            continue
        if ('\t\t' not in ground_line) and ('\t\t' not in output_line):
            assert len(ground_labels[-1])==len(output_labels[-1])
            continue

        wl_ground=ground_line.split('\t\t')
        wl_output=output_line.split('\t\t')

        ground_labels[-1].append(wl_ground[1].replace('\n',''))
        output_labels[-1].append(wl_output[1].replace('\n',''))

    assert len(ground_labels)==len(output_labels)
    # assert len(ground_labels)==len(output_lines)
    # print(len(ground_labels))
    # print(len(output_lines))

    flat_ollie_truth=[item for sublist in ground_labels for item in sublist]
    flat_ollie_predictions=[item for sublist in output_labels for item in sublist]
    assert len(flat_ollie_truth)==len(flat_ollie_predictions)
    labels=['B-Subj','I-Subj','B-Pred','I-Pred','B-Obj','I-Obj','B-enabler','I-enabler','B-attrib','I-attrib','B-time',
            'I-time','B-location','I-location','O']
    accuracy=sklearn.metrics.accuracy_score(flat_ollie_truth,flat_ollie_predictions)*100
    print('Accuracy: ',accuracy)
    precision=sklearn.metrics.precision_score(flat_ollie_truth,flat_ollie_predictions,labels=labels,
                                              average='weighted')*100
    print('Precision: ',precision)
    recall=sklearn.metrics.recall_score(flat_ollie_truth,flat_ollie_predictions,labels=labels,average='weighted')*100
    print('Recall: ',recall)
    f1_score=sklearn.metrics.f1_score(flat_ollie_truth,flat_ollie_predictions,average='weighted')*100
    print('F1 score: ',f1_score)
    f1_vect=sklearn.metrics.f1_score(flat_ollie_truth,flat_ollie_predictions,labels=labels,average=None)
    print('labels= ',numpy.asarray(labels))
    print('F1 for each label: ',f1_vect)
    conf_matrix=sklearn.metrics.confusion_matrix(flat_ollie_truth,flat_ollie_predictions,labels=labels)

    ollie_groundtruth_file.close()
    ollie_output_file.close()

    return accuracy,f1_score,conf_matrix

[acc,f1,conf_mat]=get_ollie_iob_performance('data/ollie_output.iob.txt','data/ollie-scored.iob.txt')

# print('Accuracy = ',acc)
# print('F1 score = ',f1)
