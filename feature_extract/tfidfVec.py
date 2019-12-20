from gensim import corpora
from collections import defaultdict


documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]


class mindoc2vec():
    def __init__(self):
        self.stoplist=set(["for","a","of","the","and","to","in"])
    def to_vec(self,documents):
        texts=[[word for word in document.lower().split()
                if word not in self.stoplist] for document in documents]

        frequency=defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token]+=1

        texts=[[token for token in text if frequency[token]>1] for text in texts]

        #转换成字典
        dic=corpora.Dictionary(texts)
        #dic.save('../tmp_file/deerwester.dict')
        #print(dic.token2id)

        #将新句子转换成词典向量
        # new_doc = "Human computer interaction"
        # new_vec = dic.doc2bow(new_doc.lower().split())
        # print(new_vec)
        corpus=[dic.doc2bow(text) for text in texts]
        #corpora.MmCorpus.serialize('../tmp_filr/deerwester.mm', corpus)
        #print(corpus)
        return corpus


###以上内容是全部读取到内存的，但是当数据量很大时，则内存会无法存下
class MyCorpus(object):
    def __init__(self,text_file,dictionary):
        self.my_text=text_file
        self.dic=dictionary
    def __iter__(self):
        for line in open(self.my_text):
            yield self.dic.doc2bow(line.lower().split())
#usage
# stoplist=set(["for","a","of","the","and","to","in"])
# texts=[[word for word in document.lower().split()
#                 if word not in stoplist] for document in documents]
# frequency=defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token]+=1
#
# texts=[[token for token in text if frequency[token]>1] for text in texts]
#
# #转换成字典
# dic=corpora.Dictionary(texts)
# corpus_memory_friendly = MyCorpus("my_text",dic)  # doesn't load the corpus into memory!
# #print(corpus_memory_friendly)
# for vector in corpus_memory_friendly:  # load one vector into memory at a time
#     print(vector)



#####上面字典的构建也是将全部数据加载到内存，当数据太大时，也会造成内存不够
from six import iteritems
def gen_dic(mycorpus):
    stoplist = set(["for", "a", "of", "the", "and", "to", "in"])
    # collect statistics about all tokens
    dictionary = corpora.Dictionary(line.lower().split() for line in open(mycorpus))
    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                 if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    return dictionary

#usage
my_text="my_text"
dic=gen_dic(my_text)
corpus_memory_friendly = MyCorpus(my_text,dic)  # doesn't load the corpus into memory!
#print(corpus_memory_friendly)
for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)






######转换成tfidf
from gensim import corpora, models, similarities
class tfidf_():
    def __init__(self,dictionary,corpus):
        self.dic=dictionary
        self.cor=corpus
    def tfidf_vec(self):
        tfidf = models.TfidfModel(self.cor)
        corpus_tfidf = tfidf[self.cor]
        return corpus_tfidf #是个迭代器