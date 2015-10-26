__author__ = 'heni'

import ollie_comparison.utils.preprocess_tools


def convert_to_iob(ollie_input_file, ollie_output_file):
    ollie_file = open(ollie_input_file, 'r')
    ollie_lines = ollie_file.readlines()

    iob_schema_file = open(ollie_output_file, 'w')

    for line in ollie_lines:
        line_elements=line.split('\t')
        # iob_schema_file.write(line_elements[-1])
        extractions=line_elements[2]
        spo_cont_separation=extractions.split(')(')
        spo=spo_cont_separation[0].split(';')
        try:
            assert len(spo)==3
        except:
            continue

        to_write=ollie_comparison.utils.preprocess_tools.get_iob(line_elements[-1][:-1],spo_cont_separation,is_groundtruth=True)
        iob_schema_file.write(line_elements[-1]+to_write)
        iob_schema_file.write('\n')

    ollie_file.close()
    iob_schema_file.close()


convert_to_iob('data/ollie-scored.oneExt.txt','data/ollie-scored.iob.txt')


