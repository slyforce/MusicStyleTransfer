class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.nextID = 0

        self.padID = self.add('<pad>')
        self.unkID = self.add('<unk>')
        self.bos = self.add('<s>')
        self.eos = self.add('</s>')

    def symbol(self, id):
        try:
            return self.id2word[id]
        except:
            return '<unk>'

    def id(self, word):
        try:
            return self.word2id[word]
        except:
            return self.unkID

    def add(self, word):
        if not word in self.word2id:
            self.word2id[word] = self.nextID
            self.id2word[self.nextID] = word
            self.nextID += 1
            return self.nextID - 1
        else:
            return self.word2id[word]

    def size(self):
        return self.nextID
