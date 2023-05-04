# get BERT sentence embedding from the all the sentences
# We will use the root text with minimum cleaning to bert embedding
# TDDOS - prepare preprocessing just for BERT


from sentence_transformers import SentenceTransformer
import numpy as np
from utils import get_csv, replace_enter

PATH = "./output/temp/input_data.csv"
MODEL_TYPE = "BERT"


class BERTEmbedding(object):
    def __init__(self, path, model_type):
        self.path = path
        self.model_type = model_type
        self.df = get_csv(self.path)

    def get_sentence_vec(self):
        vec = None

        self.df['bert_text'] = self.df.apply(lambda x: replace_enter(x['text']), axis=1)
        sentences = self.df['bert_text'].values.tolist()

        model = SentenceTransformer('./models')
        vec = np.array(model.encode(sentences, show_progress_bar=True))
        return vec


def main(path=PATH, model_type=MODEL_TYPE) -> None:
    bert_obj1 = BERTEmbedding(PATH, MODEL_TYPE)
    vec = bert_obj1.get_sentence_vec()
    print(vec.shape)


if __name__ == "__main__":
    main()
