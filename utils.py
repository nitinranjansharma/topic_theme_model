import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS

from wordcloud import WordCloud


def get_csv(path):
    """ Get the dataframe with indexed data"""
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return pd.DataFrame()


def replace_enter(str):
    return re.sub(r'(\n+)(?=[A-Z])', r'.', str)


def get_word_cloud(text, wc_title, wc_file_name='wordcloud.jpeg'):
    stopword_list = set(STOPWORDS)
    word_cloud = WordCloud().generate(text)
    plt.figure(figsize=(8, 6))
    plt.title(wc_title)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.savefig(wc_file_name, bbox_inches='tight')
    plt.show()
