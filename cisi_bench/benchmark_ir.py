from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
from lsi import LatentIndex
from lsi.reduction import SVD, KernelPCA, AutoEncoder
from lsi.vectorizer import BagOfWord
import argparse
import numpy as np
import dill


def get_args():
    parser = argparse.ArgumentParser(description='Information Retrieval system'
                                     ' based on Latent Semantix Indexing.\n'
                                     'This program is a demo of the  Latent'
                                     ' Semantix Indexing algorithm.')
    parser.add_argument('--embedding_dim', type=int, default=250,
                        help='Embedding dimension for latent index.')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to a pickle file containing a previously-saved LatentIndex object')
    parser.add_argument('--dump', type=str, default=None,
                        help='Path to a pickle file to save the LatentIndex object')
    parser.add_argument('--type', type=str, default='svd',
                        choices=['svd', 'kpca', 'ae'],
                        help='Dimensionality reduction technique.')
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=['poly', 'rbf', 'tanh'],
                        help='Kernel used for kernel pca,'
                             ' ignored if --type=kpca is not used.')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma term for kernel pca.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha term for kernel pca.')
    args = parser.parse_args()
    return args


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


def precision_recall_curve(relevant_documents, retrieved_top_k_docs, plot=False, save_df=None):

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
        df.to_csv(save_df + '.csv')

    if plot:
        plt.plot(recall_value, precision_value)
        plt.ylabel('precision')
        plt.xlabel('recall')
        plt.grid()
        plt.show()

    return precision_value, recall_value


reduction_dict = {'svd': SVD,
                  'kpca': KernelPCA,
                  'ae': AutoEncoder}

if __name__ == "__main__":

    args = get_args()

    # embedding dimension
    embedding_dim = args.embedding_dim  # 1200

    # top rank
    top_rank = 5000  # 20

    # corpus and queries
    corpus = pd.read_csv("cisi.csv")
    query_res = pd.read_csv("relevance.csv")
    query = pd.read_csv("query.csv")

    # building the index
    if args.load is None:
        vectorizer = BagOfWord(corpus)

        # choose reduction method
        reduction = reduction_dict[args.type]
        kwargs = {'embedding_dim': args.embedding_dim}
        if args.type == 'kpca':
            additional_kwargs = {'kernel': args.kernel,
                                 'gamma': args.gamma,
                                 'alpha': args.alpha}
            kwargs.update(additional_kwargs)
        dim_reduction = reduction(**kwargs)

        lsi = LatentIndex(corpus, vectorizer, dim_reduction)
        if args.dump is not None:
            with open(args.dump, 'wb') as f:
                dill.dump(lsi, f)
    else:
        with open(args.load, 'rb') as f:
            lsi = dill.load(f)

    # iterate over queries
    mean_ave_prec = []
    mean_precision = []
    mean_recall = []
    for _, data in query.iterrows():

        # extract query
        single_query = data['query']
        idx_query = data['query_id']

        # which document are relevat to this query?
        relevant_docs = list(query_res[query_res['query_id'] == idx_query].id)

        # extract top
        result = lsi.query(single_query, top_rank)
        retrieved_top_k_docs = list(result['id'])
        precision_value = []
        for idx in range(1, top_rank+1):
            retrieved_docs = retrieved_top_k_docs[:idx]

            # calculate precisoon
            precision_val_tmp = precision(retrieved_docs=retrieved_docs,
                                          relevant_docs=relevant_docs)
            precision_value.append(precision_val_tmp)

        mean_ave_prec.append(sum(precision_value) / len(precision_value))

        final_prec, final_recall = precision_recall_curve(relevant_docs,
                                                          retrieved_docs,
                                                          False, False)
        mean_precision.append(np.array(final_prec))
        mean_recall.append(np.array(final_recall))

    # plt.plot(np.mean(mean_recall, axis=0), np.mean(mean_precision, axis=0))
    # plt.ylabel('precision')
    # plt.xlabel('recall')
    # plt.grid()
    # plt.show()
    # print(f"MAE: {sum(mean_ave_prec)/len(mean_ave_prec)}")
