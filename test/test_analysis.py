import unittest
from itertools import combinations
import numpy as np
from scipy.stats import norm

from ccd.analysis.spotfitting import fwhm, Gauss2D, GaussContExpo
from ccd.analysis.normalize import rowwise_offset, FrameNormalizer

class TestHelperFunctions(unittest.TestCase):
    def test_fwhm(self):
        a = np.zeros(12)
        a[1:6] = np.arange(5) * 3
        a[6:11] = 3 * np.arange(5)[::-1]
        i0, i1 = fwhm(a, axis=0)
        self.assertEqual(i0, 3)
        self.assertEqual(i1, 8)


class TestNormalize(unittest.TestCase):
    def test_rowwise_offset(self):
        """Test calling rowwise_offset.

        Covered::
            * Calling directly (via __new__) or using __call__ .
            * Specifying columns as tuple or slice.
        """
        ncol = 100
        nrow = 200
        data = np.ones((nrow, ncol)) * np.random.rand(nrow)[:,np.newaxis]

        corrected = (
            # Applied directly in __new__ ...
            # ... with column-index tuple
            data - rowwise_offset((90, None), data),
            # ... with column-slice
            data - rowwise_offset(slice(90, None), data),
            # ... with column index-array
            data - rowwise_offset(range(90, 100), data),

            # Applied using __call__ ...
            # ... with column-index tuple
            data - rowwise_offset((90, None))(data),
            # ... with column-slice
            data - rowwise_offset(slice(90, None))(data),
            # ... with column index-array
            data - rowwise_offset(range(90, 100))(data),
        )

        self.assertTrue((corrected[0] < 1e-14).all())

        for a, b in combinations(corrected, 2):
            self.assertTrue((a == b).all())

    def test_normalizer(self):
        """Test FrameNormalizer

        Covered::
            * Correcting row-wise offset.
            * Extracting data region.
        """
        ncol = 100
        nrow = 200
        # data window
        data_x1, data_x2 = 10, 80
        data_y1, data_y2 = 10, 180
        # overscan columns
        overscan_cols = slice(90, None)
        # overscan offset
        offset = 3.3
        # maximum index
        x_max = 5
        y_max = 8

        normalizer = FrameNormalizer((nrow, ncol),
                                     (data_y1, data_y2),
                                     (data_x1, data_x2),
                                     rowwise_offset(overscan_cols))

        # Prepare dummy data with per-row offset
        raw_data = np.ones(ncol) * np.arange(nrow)[:, np.newaxis]
        # ... add global overscan column offset
        raw_data[:, overscan_cols] += offset
        # ... add single maximum entry
        raw_data[y_max + data_y1, x_max + data_x1] += 1e4

        offset_corrected = normalizer.correct_offset(raw_data)
        data_region = normalizer.data_region(offset_corrected)

        self.assertEqual((y_max, x_max),
                         np.unravel_index(data_region.argmax(),
                                          data_region.shape))
        rows = list(range(data_region.shape[0]))
        rows.remove(y_max)
        for y1, y2 in combinations(rows, 2):
            self.assertTrue((data_region[y1] - data_region[y2] < 1e-14).all())


class TestFitting(unittest.TestCase):
    def testGaussian2D(self):
        gauss2d = Gauss2D(0.5/np.pi, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)

        self.assertEqual(gauss2d.integral(), 1.0)
        self.assertEqual(gauss2d.evaluate(0.0, 0.0), 0.5/np.pi)

        gauss2d.update_params(mu0=-78.9)
        self.assertEqual(gauss2d.evaluate(-78.9, 0.0), 0.5/np.pi)
        self.assertGreater(gauss2d.evaluate(-78.9, 0.0), gauss2d.evaluate(-78.9, 1.0))

        gauss2d.update_params(offset=1234.5)
        self.assertEqual(gauss2d.integral(), 1.0)

        # Test the actual fitting.
        mu0 = 40.0
        mu1 = 60.0
        sigma0 = 10.0
        sigma1 = 7.0
        x = np.arange(100)
        X0, X1 = np.meshgrid(x, x, indexing="ij")
        Z = norm.pdf((X0 - mu0) / sigma0) * norm.pdf((X1 - mu1) / sigma1)

        gauss2d.initial_guess(Z)
        gauss2d.fit(Z)
        self.assertAlmostEqual(gauss2d.param_dict["mu0"], mu0)
        self.assertAlmostEqual(gauss2d.param_dict["mu1"], mu1)
        self.assertAlmostEqual(gauss2d.param_dict["sigma0"], sigma0)
        self.assertAlmostEqual(gauss2d.param_dict["sigma1"], sigma1)








if __name__ == "__main__":
    unittest.main()
