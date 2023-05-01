# Model LDA from input file in temp folder
# define params - later this needs to come from config file
from utils import get_csv
from nltk.tokenize import word_tokenize

PATH = "./output/temp/input_data.csv"
MODEL_TYPE = "LDA"
K = 10


class GetTopicsFromStatModels(object):
    def __init__(self, path, model_type='LDA', k=10) -> None:
        self.path = path
        self.model_type = model_type
        self.k = k
        self.df = get_csv(self.path)

    def lda_data_prep(self) -> None:
        sentences = list(self.df['edited_text'].values)
        print(sentences)


def main(path=PATH, model_type=MODEL_TYPE, k=K) -> None:
    lda_obj1 = GetTopicsFromStatModels(PATH, MODEL_TYPE, K)
    lda_obj1.lda_data_prep()


if __name__ == "__main__":
    main()
