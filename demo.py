from latentindex import LatentIndex
from vectorizer import BagOfWord
from reduction import SVD, KernelPCA, AutoEncoder
import os
import time
import argparse
import pandas as pd
import pickle
from pickle import load, dump
from nltk import corpus


def get_args():
    parser = argparse.ArgumentParser(description='Information Retrieval system'
                                     ' based on Latent Semantix Indexing.\n'
                                     'This program is a demo of the  Latent'
                                     ' Semantix Indexing algorithm.')
    parser.add_argument('--ndocs', type=int, default=2500,
                        help='Maximum number of documents to load from corpus')
    parser.add_argument('--rank', type=int, default=10,
                        help='Display the top rank retrievals')
    parser.add_argument('--embedding_dim', type=int, default=250,
                        help='Embedding dimension for latent index.')
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
    parser.add_argument('--load', type=str, default=None,
                        help='Path to a pickle file containing a previously-saved LatentIndex object')
    parser.add_argument('--dump', type=str, default=None,
                        help='Path to a pickle file to save the LatentIndex object')

    args = parser.parse_args()
    return args


def extract_title(text: str) -> str:
    i = 0
    while i < len(text) and text[i].upper() == text[i]:
        i += 1
    i = min(i, 40)
    return text[:i]


reduction_dict = {'svd': SVD,
                  'kpca': KernelPCA,
                  'ae': AutoEncoder}

if __name__ == "__main__":

    args = get_args()

    # creating new object if it doesn't exist
    if args.load is None:
        # creating document
        file_ids = corpus.reuters.fileids()
        data = []
        for i in range(min(args.ndocs, len(file_ids))):
            words = corpus.reuters.words(file_ids[i])
            text = " ".join(words)
            title = extract_title(text)
            data.append([file_ids[i], title, text])
        df = pd.DataFrame(data, columns=['id', 'title', 'text'])

        # choose reduction method
        reduction = reduction_dict[args.type]
        kwargs = {'embedding_dim': args.embedding_dim}
        if args.type == 'kpca':
            additional_kwargs = {'kernel': args.kernel,
                                 'gamma': args.gamma,
                                 'alpha': args.alpha}
            kwargs.update(additional_kwargs)
        reduction = reduction(**kwargs)

        # choose vectorizer method
        vectorizer = BagOfWord(df)
        latent_index = LatentIndex(df, vectorizer, reduction)
        if args.dump is not None:
            with open(args.dump, 'wb') as f:
                dump(latent_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.load, 'rb') as f:
            latent_index = load(f)

    # performing the query
    # TODO: add a nice method for printing
    while True:
        os.system('clear')
        query = input("Enter a plain-text query: ")
        start = time.time()
        results = latent_index.query(query)
        end = time.time()
        print("\nQuery time: %.2f seconds\n" % (end - start))
        print(results[["title", "text", "similarity"]].head(args.rank))

        if input("\nPress enter to continue or q to quit: ") == "q":
            break
