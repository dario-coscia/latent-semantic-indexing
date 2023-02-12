import numpy as np
import torch
from tqdm import tqdm
from abc import ABCMeta, abstractmethod


class Reduction(metaclass=ABCMeta):
    """Abstract class for reduction techniques"""

    def __init__(self, embedding_dim):
        """Abstract constructor

        :param embedding_dim: reduce dimension manifold
        :type embedding_dim: int
        """
        if isinstance(embedding_dim, int) and embedding_dim >= 1:
            self._rank = embedding_dim

        else:
            raise ValueError("expected the embedding dimension to be "
                             "an int greater than one.")

        # reduced document matrix
        self._reduced_doc_matrix = None

    def __call__(self, term_document_matrix):

        # check consistency
        if not isinstance(term_document_matrix, np.ndarray):
            raise ValueError("expected a numpy array")

        if len(term_document_matrix.shape) != 2:
            raise ValueError("expected a numpy 2 dimensional array")

        # perform optimization
        print(f"Computing term-document matrix reduction starts")
        self._fit(term_document_matrix)

    @abstractmethod
    def _fit(self, term_document_matrix):
        """
        Perform the fitting optimization

        :param X: the term document matrix
        :type X: np.array
        """
        pass

    @abstractmethod
    def from_vect_to_embedding(self, vect):
        pass

    @abstractmethod
    def from_embedding_to_vect(self, embedding):
        pass

    @property
    def embedding_dim(self):
        return self._rank


class SVD(Reduction):

    def __init__(self, embedding_dim):
        """SVD reduction technique using truncated reduction

        :param embedding_dim: dimension of the manifold embedding
        :type embedding_dim: int
        """
        super().__init__(embedding_dim=embedding_dim)

    def _fit(self, X):
        """
        Perform Truncated Singular Value Decomposition.
        :param numpy.ndarray X: the matrix to decompose.

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """

        # compute svd
        u, s, v = np.linalg.svd(X, full_matrices=False)
        s = np.diag(s)

        # reduce order matrices
        u = u[:, :self._rank]
        s = s[:self._rank, :self._rank]
        v = v

        # save u @ sigma
        self._u_times_sigma = np.dot(u, s)

        # save pseudoinverse
        inverse_s = np.linalg.pinv(s)
        tranpose_u = np.transpose(u)
        self._invsigma_times_ut = np.dot(inverse_s, tranpose_u)

        # document matrix
        self._reduced_doc_matrix = self.from_vect_to_embedding(X)

    def from_vect_to_embedding(self, vect):
        """
        Return the embedding given a vector

        :param vect: the vector to be embedded
        :type vect: numpy.ndarray
        :return: the embedding
        :rtype: numpy.ndarray
        """
        return np.dot(self._invsigma_times_ut, vect)

    def from_embedding_to_vect(self, embedding):
        """
        Return the vector given the embedding

        :param vect: the embedded vector
        :type vect: numpy.ndarray
        :return: the vector in original space
        :rtype: numpy.ndarray
        """
        return np.dot(self._u_times_sigma, embedding)


class KernelPCA(Reduction):

    def __init__(self, embedding_dim, kernel, gamma=1.0, alpha=1.0):
        """SVD reduction technique using truncated reduction

        :param embedding_dim: dimension of the manifold embedding
        :type embedding_dim: int
        :param kernel: pca kernel {'poly', 'rbf', 'tanh'} possible option.
        :type kernel: string
        :param gamma: coefficient for rbf, poly kernel; ignored by others.
        :type gamma: int
        """
        super().__init__(embedding_dim=embedding_dim)

        # list of possible kernels
        possible_kernels = {'poly', 'rbf', 'tanh'}

        if kernel not in possible_kernels:
            raise ValueError(
                "expected kernel to be in list of possible kernels.")
        else:
            self._kernel = self._choose_kernel(kernel)

        # gamma coefficient + alpha coefficient
        self._gamma = gamma
        self._alpha = alpha

    def _choose_kernel(self, kernel_type):

        def rbf(X):
            X_norm = np.einsum('ij,ij->i', X, X)
            K0 = (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
            K = np.exp(-self._gamma * K0)
            return K

        def poly(X):
            K0 = np.dot(X, X.T)
            return np.power(K0, self._gamma)

        def tanh(X):
            K0 = np.dot(X, X.T)
            return np.tanh(K0)

        kernels = {'rbf': rbf,
                   'poly': poly,
                   'tanh': tanh}

        return kernels[kernel_type]

    def _fit(self, X):
        (m, _) = X.shape

        # creating kernel matrix
        K = self._kernel(X)

        # centering matrix
        center_coeff = np.eye(m) - np.eye(m) / m
        K = np.dot(np.dot(center_coeff, K), center_coeff)

        # performing svd
        u, s, v = np.linalg.svd(K, full_matrices=False)
        s = np.sqrt(np.diag(s))

        # reduce order matrices
        u = u[:, :self._rank]
        s = s[:self._rank, :self._rank]

        # save u @ sigma, document reduce matrix
        self._u_times_sigma = np.dot(u, s)

        # compute pseudo_inverse using ridge regression
        pinv_s = np.linalg.inv(
            np.dot(s.T, s) - self._alpha * np.eye(s.shape[0]))
        pinv_s = np.dot(pinv_s, s.T)
        tranpose_u = np.transpose(u)
        self._invsigma_times_ut = np.dot(pinv_s, tranpose_u)

        # save document recuced matrix
        self._reduced_doc_matrix = self.from_vect_to_embedding(X)

    def from_vect_to_embedding(self, vect):
        """
        Return the embedding given a vector

        :param vect: the vector to be embedded
        :type vect: numpy.ndarray
        :return: the embedding
        :rtype: numpy.ndarray
        """
        return np.dot(self._invsigma_times_ut, vect)

    def from_embedding_to_vect(self, embedding):
        """
        Return the vector given the embedding

        :param vect: the embedded vector
        :type vect: numpy.ndarray
        :return: the vector in original space
        :rtype: numpy.ndarray
        """
        return np.dot(self._u_times_sigma, embedding)


class AutoEncoder(Reduction):

    def __init__(self, embedding_dim, encoder=None, decoder=None, epochs=50,
                 optimizer=torch.optim.Adam, optimizer_kwargs=None,
                 lr=0.001, batch_size=24):
        """AutoEncoder dimensionality redution.

        :param embedding_dim: dimension of the manifold embedding
        :type embedding_dim: int
        :param encoder: encoder network (output dim = embedding_dim)
        :type encoder: torch.nn.Module
        :param decoder: encoder network (input dim = embedding_dim)
        :type decoder: int
        :param int epochs: number of epochs to optimize the AutoEncoder
        :param torch.optim optimizer: the neural network optimizer to use;
            default is `torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param float lr: the learning rate; default is 0.001.

        Note:
        No checks are done to ensure compatibility encoder decoder.
        Default MSE loss is minimized.
        """
        super().__init__(embedding_dim=embedding_dim)

        # check encoder/decoder are torch modules
        self._check_consistency(encoder, decoder)

        self._encoder = encoder
        self._decoder = decoder
        self._loss = torch.nn.MSELoss()

        if not isinstance(batch_size, int):
            raise ValueError("batch_size must be int")
        else:
            self._batch_size = batch_size

        if not isinstance(epochs, int):
            raise ValueError("epoch must be int")
        else:
            self._iter = epochs

        if not optimizer_kwargs:
            self._optimizer_kwargs = {}
        self._optimizer_kwargs['lr'] = lr
        self._optimizer = optimizer

    def _handle_dtype(self, input_):
        """
        Perform dytpe conversion.

        :param np.ndarray input_: input for dtype torch conversion
        """

        # conver input into torch tensor
        X = torch.from_numpy(input_)

        # only if we don't use floats
        if input_.dtype != 'float':
            dtype = torch.float
            X = X.to(dtype)

        # change model dtype
        self._model = self._model.to(X.dtype)

        return X

    def _fit(self, X):
        """
        Perform Training for Dimensionality Reduction.
        """
        (n_terms, n_docs) = X.shape
        if self._rank > n_docs:
            self._rank = n_docs
        X = X.T

        if self._encoder is None:
            self._encoder = torch.nn.Sequential(
                torch.nn.Linear(n_terms, n_terms // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(n_terms // 2, n_terms // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(n_terms // 4, self._rank)
            )

        if self._decoder is None:
            self._decoder = torch.nn.Sequential(
                torch.nn.Linear(self._rank, n_terms // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(n_terms // 4, n_terms // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(n_terms // 2, n_terms)
            )
        self._model = torch.nn.Sequential(self._encoder, self._decoder)
        self._optimizer = self._optimizer(
            self._model.parameters(), **self._optimizer_kwargs)

        X = self._handle_dtype(X)

        # batching

        dataloader = torch.utils.data.DataLoader(
            X, batch_size=self._batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(device)

        self._model.train()
        for _ in tqdm(range(self._iter)):
            for x in dataloader:
                x = x.to(device)
                output = self._model(x)
                loss = self._loss(x, output)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

        X = X.to('cpu')
        self._model = self._model.to('cpu')
        self._model.eval()

        # save document recuced matrix
        self._reduced_doc_matrix = self._encoder(X).detach().numpy().T

    def _check_consistency(self, encoder, decoder):
        """Checking consistency encoder decoder structure.
        """

        if encoder is not None:
            if not isinstance(encoder, torch.nn.Module):
                raise ValueError(
                    "expected encoder to be instance of torch.nn.Module class")

        if decoder is not None:
            if not isinstance(decoder, torch.nn.Module):
                raise ValueError(
                    "expected decoder to be instance of torch.nn.Module class")

    def from_vect_to_embedding(self, vect):
        """
        Return the embedding given a vector.

        :param vect: the vector to be embedded
        :type vect: numpy.ndarray
        :return: the embedding
        :rtype: numpy.ndarray
        """
        vect = self._handle_dtype(vect)
        vect = self._encoder(vect).detach()
        return vect.numpy()

    def from_embedding_to_vect(self, embedding):
        """
        Return the vector given the embedding

        :param vect: the embedded vector
        :type vect: numpy.ndarray
        :return: the vector in original space
        :rtype: numpy.ndarray
        """
        vect = self._handle_dtype(embedding)
        vect = self._decoder(vect).detach()
        return vect.numpy()
