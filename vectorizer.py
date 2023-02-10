import pandas as pd
import re
import numpy as np
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

STOPWORDS = set(stopwords.words('english'))


class Vectorizer(metaclass=ABCMeta):
    def __init__(self, corpus: pd.DataFrame):
        """Vetcorizer base class

        :param corpus: corpus dataframe with 'text' column
        :type corpus: pd.DataFrame
        """

        if isinstance(corpus, pd.DataFrame):
            if 'text' not in corpus.columns:
                raise ValueError(
                    "'text' expected to be a column of the dataframe")
            self._corpus = corpus
            self._corpus['original_text'] = corpus['text']
            self._corpus['DocId'] = corpus.index
        else:
            raise ValueError(
                'expected corpus to be a dataframe with entry text')

        # preprocess
        tqdm.pandas(desc='{:<25}'.format('Preprocessing corpus'))
        self._corpus["text"] = self._corpus["text"].progress_apply(
            self._preprocess_text)

        # build vocabulary
        tqdm.pandas(desc='{:<25}'.format('Building vocabulary'))
        self._vocabulary = self._build_vocabulary()

        # for document matrix
        self._number_terms = len(self._vocabulary)
        self._number_documents = len(self._corpus['DocId'])

    @staticmethod
    def _preprocess_text(text):
        """
        Preprocess each string by:
        1. Lowercasing the string
        2. Removing punctuation and other non-alphanumeric characters
        3. Lemmatizing the string using WordNet's lemmatizer
        4. Removing stopwords (assumed to be English)

        params:
            text (str): string to preprocess
        returns:
            text (str): preprocessed string

        Credits: Alessandro Pierro https://github.com/AlessandroPierro/BM25
        """
        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9]', " ", text)
        text = re.sub(r'[0-9]+', '', text)
        text = " ".join([WordNetLemmatizer().lemmatize(word)
                         for word in text.split(" ")])
        text = " ".join([word for word in text.split(" ")
                         if word not in STOPWORDS])
        return text

    def _build_vocabulary(self):
        """
        Returns a set of all unique words in the corpus, based on
        the 'text' column of self._corpus.

        returns:
            vocabulary (set): set of all unique words in the corpus

        Adapted from: https://github.com/AlessandroPierro/BM25
        """
        vocabulary = set()
        self._corpus["text"].progress_apply(
            lambda text: vocabulary.update(text.split(" ")))
        vocabulary.discard("")
        return list(vocabulary)

    @abstractmethod
    def create_term_document_matrix(self):
        pass


class BagOfWord(Vectorizer):
    def __init__(self, corpus: pd.DataFrame):
        """Bag Of Word

        :param corpus: corpus dataframe with 'text' column
        :type corpus: pd.DataFrame
        """
        super().__init__(corpus)

    def create_term_document_matrix(self):
        matrix = np.empty(shape=(self._number_terms,
                                 self._number_documents))
        df = pd.DataFrame(self._corpus['text'])
        for idx_term, term in tqdm(enumerate(self._vocabulary),
                                   desc='{:<25}'.format(
                                       "Computing term-document matrix"),
                                   total=self._number_terms):
            count = df.apply(lambda x: sum(
                [i.count(term) for i in x if isinstance(i, str)]),
                axis=1).to_numpy()
            matrix[idx_term, :] = count

        return matrix

    def query(self, query):
        query = self._preprocess_text(query)
        query_terms = set(query.split(" "))

        query_array = np.zeros(shape=(self._number_terms,))
        for idx_term, term in enumerate(self._vocabulary):
            if term in query_terms:
                query_array[idx_term] = 1.

        if np.linalg.norm(query_array) == 0.:
            print("No match found... Try to change a bit the query")
            return None

        return query_array
