# Get the clusters from vectors created
# TODOS - combining and make provisions for experimentation
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from BERT_embedding import BERTEmbedding
from LDA import GetTopicsFromStatModels
from train_AE import TrainAE

PATH = "./output/temp/input_data.csv"
INPUT_SHAPE = 773  # 768+5
MODEL_TYPE = 'BERT'


class GetCluster(object):
    def __init__(self, vec, method='kmeans'):
        self.method = method
        self.vec = vec

    def fit_cluster(self):
        if self.method == 'kmeans':
            # self.vec = self.vec.detach().numpy()
            kmeans = KMeans(n_clusters=3, init='k-means++')
            kmeans.fit(self.vec)
            return kmeans.predict(self.vec)
        else:
            # TODO implement exception and other strategies
            return None


def main():
    bert_obj1 = BERTEmbedding(PATH, model_type='BERT')
    vec_bert = bert_obj1.get_sentence_vec()
    lda_obj1 = GetTopicsFromStatModels(PATH, model_type='LDA', k=5)
    lda_obj1.lda_data_prep()
    lda_model, corpus = lda_obj1.run_lda()
    lda_vec = lda_obj1.get_lda_vec(lda_model, corpus)
    lda_vec = lda_vec.astype(float)

    vec = np.concatenate((vec_bert, lda_vec), axis=1)
    training_obj = TrainAE(PATH, vec)
    reduced = training_obj.model.encoder(torch.from_numpy(vec).float())
    reduced = reduced.detach().numpy()

    get_cluster = GetCluster(vec=reduced, method='kmeans')
    cluster_idx = get_cluster.fit_cluster()
    print(cluster_idx)
    # get the temp csv and write back to it
    df = pd.read_csv(PATH)
    df['cluster_idx'] = cluster_idx
    df.to_csv(PATH)


if __name__ == '__main__':
    main()
