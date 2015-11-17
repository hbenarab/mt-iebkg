__author__ = 'heni'


# this class allows to manage word-to-index dictionaries
class WordEmbeddings(object):
    def __init__(self):
        self.words = []
        self.words2index = {}
        self.index2words = {}
        self.changed = False

    def setDictionary(self,dict):
        self.words2index=dict
        return self.words2index

    # returns the size of the dictionary
    def getSize(self):
        return len(self.words)

    # returns current word-to-index dictionary
    def getCurrentIndex(self):
        return self.words2index

    # returns index-to-word dictionary
    def getIndex2Word(self):
        if self.changed or self.index2words is None:
            # Recalculate inverse index
            self.index2words = {}
            for w in self.words2index:
                self.index2words[self.words2index[w]] = w
        return self.index2words

    # load a dictionary and merge it with the current one
    def loadIndex(self, indexPath):
        otherIndex = pickle.load(open(indexPath, 'r'))
        self.merge(otherIndex)

    # flag corresponding to the status of the dictionary (changed or not)
    def setChanged(self):
        self.changed = True

    # replace current dictionary with another one
    def loadIndexReplace(self, indexPath):
        self.words2index = pickle.load(open(indexPath, 'r'))
        self.setChanged()

    # add words to the current dictionary
    def addWords(self, words):
        currentMax = len(self.words2index)
        for w in words:
            if not w in self.words2index:
                self.words2index[w] = currentMax
                currentMax += 1
        self.setChanged()

    # add A SINGLE WORD to the current dictionary
    def add_word(self, word):
        currentMax = len(self.words2index)
        # if currentMax==0:
        #     currentMax=1

        if not word in self.words2index:
            self.words2index[word] = currentMax+1
            currentMax += 1
        self.setChanged()

    # add words in input sentences to the current dictionary
    def addSentences(self, sentences):
        for s in sentences:
            words = s.split()
            self.addWords(words)
        self.setChanged()

    # merge input dictionary with the current one
    def merge(self, otherIndex):
        currentMax = len(self.words2index)
        for word in otherIndex.words2index:
            if not word in self.words2index:
                self.words2index[word] = currentMax
                currentMax += 1
        self.setChanged()

    # store current dictionary in a Pickle file
    def storeCurrentIndex(self, indexPath):
        pickle.dump(self.words2index, open(indexPath, 'wb'))


# create word-to-index dictionary
def create_word_index(sentences):
    index = 0
    words2index = {}
    for s in sentences:
        # s = s[0]
        words = s.split()
        for w in words:
            if w not in words2index:
                words2index[w] = index
                index += 1

    return words2index
