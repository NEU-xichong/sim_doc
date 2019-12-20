import smart_open
import os
import gensim
# Set file names for train and test data

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

my_text="my_text"
train_corpus = list(read_corpus(my_text))


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
#model = Doc2Vec(documents, dm=1, size=100, window=8, min_count=5, workers=4)
model.build_vocab(train_corpus)

# model.save('models/ko_d2v.model')
# model = gensim.models.doc2vec.Doc2Vec.load('models/ko_d2v.model')

doc="The intersection graph of paths in trees"
vector = model.infer_vector(doc.split())
print(vector)