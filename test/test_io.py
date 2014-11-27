import unittest
import numpy as np

from ccd.io import frameset


class TestFrameSet(unittest.TestCase):
    def test_masked(self):
        """Test than FrameSet.mean_frame respects masked arrays."""
        shape = (2, 2)
        d1 = 1.0 * np.ones(shape)
        d2 = 2.0 * np.ones(shape)
        d3 = np.ma.masked_array(3.0 * np.ones(shape), mask=[[0,0], [1,1]])

        fs = frameset.FrameSet([d1, d2, d3])
        m = fs.mean_frame()
        self.assertEqual(m[0, 0], 2.0)
        self.assertEqual(m[0, 1], 2.0)
        self.assertEqual(m[1, 0], 1.5)
        self.assertEqual(m[1, 1], 1.5)

if __name__ == "__main__":

    unittest.main()
