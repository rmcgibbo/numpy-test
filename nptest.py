from numpy.testing import assert_array_almost_equal
from numpy import linalg, arange, float64, array, dot, transpose
import numpy as np

np.__config__.show()

def test_eig_vs_eigh_above_560():
        # gh-6896
        N = 560

        A = np.arange(N*N).reshape(N, N)
        A = A + A.T

        w1 = np.sort(linalg.eig(A)[0])
        w2 = np.sort(linalg.eigh(A, UPLO='U')[0])
        w3 = np.sort(linalg.eigh(A, UPLO='L')[0])
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(w1, w3)

test_eig_vs_eigh_above_560()
print('Finished')
