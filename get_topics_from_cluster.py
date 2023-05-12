# get topics from clusters index
# https://www.kaggle.com/code/panks03/clustering-with-topic-modeling-using-lda
# https://stackoverflow.com/questions/59354365/how-to-extract-topics-from-existing-text-clusters
# https://github.com/topics/text-clustering?o=asc&s=stars

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

PATH = "./output/temp/input_data.csv"
n = 5


class GetImpWords:
    def __init__(self, PATH):
        self.path = PATH
        self.df = pd.read_csv(self.path)

    def __len__(self):
        return len(self.df)

    def return_top_words(self, clus_idx, n=5):
        vec = TfidfVectorizer()
        subset = self.df.loc[self.df['cluster_idx'] == clus_idx]
        docs = subset['edited_text'].tolist()
        x = vec.fit_transform(docs)
        feature_array = np.array(vec.get_feature_names())
        tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]
        top_n = feature_array[tfidf_sorting][:n]
        return top_n


def main():
    get_words = GetImpWords(PATH)
    for idx in get_words.df['cluster_idx'].unique():
        top_n = get_words.return_top_words(idx, n)
        print(" Cluster --- {}".format(idx))
        print(top_n)


if __name__ == "__main__":
    main()
