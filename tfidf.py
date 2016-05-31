#encoding=utf-8
import math
import re
# from word_tokenizer import PortugueseWordTokenizer

class TFIDF(object):
    """docstring for TFIDF"""
    def __init__(self, corpus_dict, stopword_filename = None, DEFAULT_IDF = 1.5):
        super(TFIDF, self).__init__()
        self.num_docs = 0
        self.term_num_docs = {}     # term : num_docs_containing_term
        self.stopwords = set([])
        self.idf_default = DEFAULT_IDF
        # self._tokenizer = PortugueseWordTokenizer()

        if not corpus_dict:
            print "corpus is empty!"
            exit()
        self.num_docs = len(corpus_dict)
        # self.corpus = list(corpus_dict)
        self.corpus = [self.getTokens(doc) for doc in corpus_dict]
        if stopword_filename:
            stopword_file = codecs.open(stopword_filename, "r", encoding='utf-8')
            self.stopwords = set([line.strip() for line in stopword_file])

    def getTokens(self,string):
        # return self._tokenizer.tokenize(string)
        return re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", string.lower())

    def getTf(self,innerIndexes):
        wordFrequence = {}
        wordCount = 0
        for doc in innerIndexes:
            for oneToken in self.corpus[doc]:
                count = wordFrequence.setdefault(oneToken,0) + 1
                wordFrequence[oneToken] = count
                wordCount += 1
        for index,value in wordFrequence.iteritems():
            wordFrequence[index] = float(value)/wordCount
        return wordFrequence

    def getTermDocs(self):
        for oneAricles in self.corpus:
            # articleWordSet = set(self.getTokens(oneAricles))
            # for word in articleWordSet:
            for word in set(oneAricles):
                articles = self.term_num_docs.get(word,0) + 1
                self.term_num_docs[word] = articles

    def getIdf(self):
        self.getTermDocs()
        wordIdf = {}
        for term,value in self.term_num_docs.iteritems():
            if term in self.stopwords:
                wordIdf[term] = 0.0
                continue
            wordIdf[term] =  math.log(float(self.num_docs + 1) / (value + 1))
        return wordIdf

    @staticmethod
    def getTfIdf(tfDict,idfDict):
        resultList = [(key,tfDict[key]*idfDict[key]) for key in tfDict]
        return sorted(resultList,key=lambda x:x[1],reverse=True)