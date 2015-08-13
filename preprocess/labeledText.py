__author__ = 'heni'


class LabeledText(object):
    def __init__(self):
        self.data = [[], []]

    def getData(self):
        return self.data

    def getSentences(self):
        return self.data[0]

    def getLabels(self):
        return self.data[1]

    def addPair(self, sentence, labels):
        if len(sentence) is not len(labels):
            raise Exception('Expecting length of given pair to be equal,'
                            'but sentence contains {0} words and labels {1} entries.'
                            .format(len(sentence), len(labels)))
        self.data[0].append(sentence)
        self.data[1].append(labels)

    def addData(self, labeledData):
        if len(labeledData) is not 2:
            raise Exception('Expecting labeled data to contain exactly two lists (with corresponding pairs)')
        if len(labeledData[0]) != len(labeledData[1]):
            raise Exception('Given lists of sentences and associated labels do not match in length.')
        self.data[0].extend(labeledData[0])
        self.data[1].extend(labeledData[1])
