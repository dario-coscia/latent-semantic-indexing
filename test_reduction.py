from reduction import SVD, KernelPCA
import numpy as np

M = np.array([[1., 0, 0, 0, 2],
              [0, 0, 3, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 2, 0, 0, 0]])


def test_svd():
    # costructor
    svd = SVD(3)
    # perform svd
    svd(M)
    # compute embedding
    embedding = svd.from_vect_to_embedding(M)
    assert embedding.shape == (3, M.shape[1])

    # reconstruct
    reconstruct = svd.from_embedding_to_vect(embedding)
    assert reconstruct.shape == M.shape
    np.testing.assert_array_equal(M, reconstruct)


def test_kpca():

    def testingkernel(kernel):
        # costructor
        kpca = KernelPCA(3, kernel, alpha=0.)
        # perform svd
        kpca(M)
        # compute embedding
        embedding = kpca.from_vect_to_embedding(M)
        assert embedding.shape == (3, M.shape[1])

        # reconstruct
        reconstruct = kpca.from_embedding_to_vect(embedding)
        assert reconstruct.shape == M.shape

    for kernel in ['poly', 'rbf', 'tanh']:
        testingkernel(kernel)
