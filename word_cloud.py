# get word cloud from whole clustered text
import pandas as pd

from utils import get_word_cloud

PATH = "./output/temp/input_data.csv"


class WordCloudClass():
    def __init__(self, path=PATH):
        self.path = path
        self.df = pd.read_csv(self.path)

    def save_word_cloud(self):
        for idx in self.df['cluster_idx'].unique():
            subset = self.df.loc[self.df['cluster_idx'] == idx]
            text = "".join(subset['edited_text'].tolist())
            get_word_cloud(text, str(idx), str(idx) + '.jpeg')


def main():
    wc = WordCloudClass(path=PATH)
    wc.save_word_cloud()


if __name__ == "__main__":
    main()
