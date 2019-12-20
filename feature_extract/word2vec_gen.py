from gensim.test.utils import datapath
from gensim import utils

class MyCorpus(object):

    def __init__(self,my_text):
        self.my_text=my_text
    def __iter__(self):
        #corpus_path = datapath(self.my_text)
        for line in open(self.my_text):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


import gensim.models


my_text="my_text"
sentences = MyCorpus(my_text)
model = gensim.models.Word2Vec(sentences=sentences,size=10,min_count=2)
print(model['interface'])