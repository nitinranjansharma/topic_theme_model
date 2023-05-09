# Get the clusters from vectors created
# TODOS - combining and make provisions for experimentation
import pandas as pd
import torch
from sklearn.cluster import KMeans
from BERT_embedding import BERTEmbedding
from train_AE import TrainAE

PATH = "./output/temp/input_data.csv"
INPUT_SHAPE = 768
MODEL_TYPE = 'BERT'


class GetCluster(object):
    def __init__(self, vec, method='kmeans'):
        self.method = method
        self.vec = vec

    def fit_cluster(self):
        if self.method == 'kmeans':
            self.vec = self.vec.detach().numpy()
            kmeans = KMeans(n_clusters=3, init='k-means++')
            kmeans.fit(self.vec)
            return kmeans.predict(self.vec)
        else:
            # TODO implement exception and other strategies
            return None


def main():
    bert_obj1 = BERTEmbedding(PATH, MODEL_TYPE)
    vec = bert_obj1.get_sentence_vec()
    training_obj = TrainAE(PATH, vec)
    output = training_obj.auto_encoder_fit()
    reduced = training_obj.model.encoder(torch.from_numpy(vec))
    get_cluster = GetCluster(vec=reduced, method='kmeans')
    cluster_idx = get_cluster.fit_cluster()
    print(cluster_idx)
    # get the temp csv and write back to it
    df = pd.read_csv(PATH)
    df['cluster_idx'] = cluster_idx
    df.to_csv(PATH)


if __name__ == '__main__':
    main()
