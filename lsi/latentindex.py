import os
import numpy as np
from .reduction import Reduction, SVD, KernelPCA
import pandas as pd
from .vectorizer import Vectorizer, BagOfWord


class LatentIndex(object):

    def __init__(self, corpus: pd.DataFrame,
                 vectorizer: Vectorizer, dim_reduction: Reduction):

        # set the vectorizer
        self._vectorizer = vectorizer

        # create matrix to reduce
        matrix = self._vectorizer.create_term_document_matrix()

        # fitting and compute reduce model
        self._dim_reduction = dim_reduction
        self._dim_reduction(matrix)

        # delete memory space
        del matrix

    def _compute_similarity(self, query_embedding):
        doc_matrix = self._dim_reduction._reduced_doc_matrix
        p1 = query_embedding.dot(doc_matrix)
        p2 = np.linalg.norm(doc_matrix, axis=0) * \
            np.linalg.norm(query_embedding)
        return p1 / p2

    def query(self, query: list[str], rank: int = None):
        query_vector = self._vectorizer.query(query)

        if query_vector is None:
            return self._vectorizer._corpus.head()

        query_vector_embedding = self._dim_reduction.from_vect_to_embedding(
            query_vector)

        # extract docIDs
        idx_similarity = self._compute_similarity(query_vector_embedding)

        # df
        df = self._vectorizer._corpus
        df["similarity"] = idx_similarity
        df["text"] = self._vectorizer._corpus["original_text"]

        if rank is None:
            return df.sort_values('similarity', ascending=False)

        return df.sort_values('similarity', ascending=False).head(rank)
