from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
from ..latentindex import LatentIndex


def intersect_docs(retrieved_docs: Union[set, list],
                   relevant_docs: Union[set, list]):

    if isinstance(retrieved_docs, list):
        retrieved_docs = set(retrieved_docs)

    if isinstance(relevant_docs, list):
        relevant_docs = set(relevant_docs)

    return relevant_docs.intersection(retrieved_docs)


def precision(retrieved_docs: Union[set, list],
              relevant_docs: Union[set, list]):

    intersection = intersect_docs(retrieved_docs=retrieved_docs,
                                  relevant_docs=relevant_docs)

    return len(intersection) / len(retrieved_docs)


def recall(retrieved_docs: Union[set, list],
           relevant_docs: Union[set, list]):

    intersection = intersect_docs(retrieved_docs=retrieved_docs,
                                  relevant_docs=relevant_docs)

    return len(intersection) / len(relevant_docs)


def precision_recall_curve(relevant_documents, retrieved_top_k_docs, plot=False, save_df=False):

    # top_k docs
    top_k = len(retrieved_top_k_docs)

    # initialization
    precision_value = []
    recall_value = []

    # calculate recall and precision
    for idx in range(1, top_k+1):
        retrieved_docs = retrieved_top_k_docs[:idx]

        # calculate precisoon
        precision_val_tmp = precision(retrieved_docs=retrieved_docs,
                                      relevant_docs=relevant_documents)
        precision_value.append(precision_val_tmp)

        # calculate recall
        recall_val_tmp = recall(retrieved_docs=retrieved_docs,
                                relevant_docs=relevant_documents)
        recall_value.append(recall_val_tmp)

    if save_df:
        df = pd.DataFrame(list(zip(precision_value, recall_value)),
                          columns=['precision', 'recall'])
        df.to_csv('results.csv')

    if plot:
        plt.plot(recall_value, precision_value)
        plt.ylabel('precision')
        plt.xlabel('recall')
        plt.grid()
        plt.show()
