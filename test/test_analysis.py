import unittest
from itertools import combinations
import numpy as np

from ccd.analysis.spotfitting import fwhm
from ccd.analysis.normalize import rowwise_offset

class TestHelperFunctions(unittest.TestCase):
    def test_fwhm(self):
        a = np.zeros(12)
        a[1:6] = np.arange(5) * 3
        a[6:11] = 3 * np.arange(5)[::-1]
        i0, i1 = fwhm(a, axis=0)
        self.assertEqual(i0, 3)
        self.assertEqual(i1, 8)


class TestNormalizer(unittest.TestCase):
    def test_rowwise_offset(self):
        data = np.ones((100, 100)) * np.random.rand(100)[:,np.newaxis]

        c1 = data - rowwise_offset((90, None), data)
        c2 = data - rowwise_offset(slice(90, None), data)
        c3 = data - rowwise_offset(slice(90, None))(data)
        c4 = data - rowwise_offset((90, None))(data)

        self.assertTrue((c1 < 1e-14).all())

        for a, b in combinations((c1, c2, c3, c4), 2):
            self.assertTrue((a == b).all())


if __name__ == "__main__":
    unittest.main()
