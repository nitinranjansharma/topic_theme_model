import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
import gensim


class Preprocessing(object):
    def __init__(self, path):
        self.path = path
        self.df = pd.DataFrame()

    def get_csv(self) -> None:
        """ Get the dataframe with indexed data"""
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)

    def __text_cleaning_helper(self, sub_str) -> str:
        """ basic text cleaning"""
        sub_str = re.sub(r'([a-z])([A-Z])', r'\1\. \2', sub_str)
        sub_str = sub_str.lower()
        # removal of < > in html
        sub_str = re.sub(r'&gt|&lt', ' ', sub_str)
        # remove multi occurences in letters (more than 2 repeating letters
        sub_str = re.sub(r'([a-z])\1{2,}', r'\1', sub_str)
        # removing asterisk in string delim
        sub_str = re.sub(r'\*|\W\*|\*\W', '. ', sub_str)
        sub_str = ' '.join([i for i in sub_str.split(" ") if i.isalpha() if len(i) > 1])
        return sub_str

    def __stemmer(self, sub_str) -> str:
        """ Only use if it is required and as mentioned in config
        can add spell corrector post stemming to rectify over stemming
        """
        ps = PorterStemmer()
        words = word_tokenize(sub_str)
        sub_str = reduce(lambda x, y: x + " " + ps.stem(y), words, "")
        return sub_str

    def text_clean(self) -> None:
        """using helper to clean text"""
        self.df['edited_text'] = self.df.apply(lambda x: self.__text_cleaning_helper(x['text']), axis=1)

    def remove_stopwords(self) -> None:
        """Default language is English here"""
        stopwords_en = stopwords.words('english')
        self.df['edited_text'] = self.df.apply(lambda x: ' '.join([i for i in x['edited_text'].split(" ")
                                                                   if i not in stopwords_en]), axis=1)

    def stemming(self) -> None:
        """Default language is English here"""
        stopwords_en = stopwords.words('english')
        self.df['edited_text'] = self.df.apply(lambda x: self.__stemmer(x['edited_text']), axis=1)

    def write_back(self) -> None:
        """ Write csv back to temp folder"""
        self.df.to_csv("./output/temp/input_data.csv", index=False)

    def sequence(self):
        self.get_csv()
        self.text_clean()
        self.remove_stopwords()
        self.stemming()
        self.write_back()


PATH = "./output/temp/input_data.csv"


def main():
    # nltk.download('stopwords')
    # nltk.download('punkt')
    prep_obj = Preprocessing(PATH)
    prep_obj.sequence()


if __name__ == "__main__":
    main()
