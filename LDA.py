# Model LDA from input file in temp folder
# define params - later this needs to come from config file
# get coherence score and iterate to get the best value of K
import numpy as np

from utils import get_csv
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from pprint import pprint
from gensim.models import LdaModel

PATH = "./output/temp/input_data.csv"
MODEL_TYPE = "LDA"
K = 5


class GetTopicsFromStatModels(object):
    def __init__(self, path, model_type='LDA', k=10) -> None:
        self.path = path
        self.model_type = model_type
        self.k = k
        self.df = get_csv(self.path)

    def lda_data_prep(self):
        sentences = self.df['edited_text'].values.tolist()
        docs = [[token for token in doc.split(' ') if token != ''] for doc in sentences]
        dictionary = corpora.Dictionary(docs)

        dictionary.filter_extremes(no_below=20, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in docs]

        return dictionary, corpus

    def run_lda(self):
        dictionary, corpus = self.lda_data_prep()

        num_topics = self.k
        chunksize = 2000
        passes = 20
        iterations = 400
        eval_every = None  # Not to evaluate model perplexity,
        # Make an index to word dictionary.
        temp = dictionary[0]
        id2word = dictionary.id2token
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )
        top_topics = lda_model.top_topics(corpus)
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print('Average topic coherence: %.4f.' % avg_topic_coherence)

        return lda_model, corpus

    def get_lda_vec(self, lda_model, corpus):
        """ Get vectorize results from LDA"""
        num_doc = len(corpus)
        vec_lda = np.zeros((num_doc, self.k))
        for i in range(num_doc):
            # get the distribution for the i-th document in corpus
            for topic, prob in lda_model.get_document_topics(corpus[i]):
                vec_lda[i, topic] = prob
        return vec_lda


def main(path=PATH, model_type=MODEL_TYPE, k=K) -> None:
    lda_obj1 = GetTopicsFromStatModels(PATH, MODEL_TYPE, K)
    lda_obj1.lda_data_prep()
    lda_model, corpus = lda_obj1.run_lda()
    lda_vec = lda_obj1.get_lda_vec(lda_model, corpus)
    print(lda_vec.shape)
    print(" Vectors generated")


if __name__ == "__main__":
    main()
