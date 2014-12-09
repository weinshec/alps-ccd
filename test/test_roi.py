import unittest
import numpy as np

from ccd.analysis import roi



class ROIBasicTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((5, 5), dtype=np.bool)
        self.mask[[2, 3, 4], [2, 2, 2]] = True
        self.roi = roi.IrregularROI(self.mask)

    def test_shape(self):
        self.assertTrue(self.roi.shape == (5, 5))

    def test_mask(self):
        self.assertTrue((self.mask == self.roi.mask).all())

    def test_n_px(self):
        self.assertTrue(self.roi.n_px == 3)

    def test_ravelled(self):
        self.assertTrue((self.roi.ravelled == np.asarray([12, 17, 22])).all())

    def test_yx(self):
        y, x = self.roi.yx
        self.assertTrue((y == np.asarray([2, 3, 4])).all())
        self.assertTrue((x == np.asarray([2, 2, 2])).all())

    def test_xy(self):
        x, y = self.roi.xy
        self.assertTrue((y == np.asarray([2, 3, 4])).all())
        self.assertTrue((x == np.asarray([2, 2, 2])).all())

    def test_inner_idx(self):
        t = self.roi.inner_idx(0, "ravelled")
        self.assertTrue(len(t) == 0)

        t = self.roi.inner_idx(12, "ravelled")
        self.assertTrue(len(t) == 1)
        self.assertTrue((t == [0]).all())

        t = self.roi.inner_idx([12, 22], "ravelled")
        self.assertTrue(len(t) == 2)
        self.assertTrue((t == [0, 2]).all())

    def test_outer_idx(self):
        self.assertEqual(self.roi.outer_idx(0, "ravelled"), 5*2 + 2)
        self.assertEqual(self.roi.outer_idx(1, "ravelled"), 5*3 + 2)
        self.assertEqual(self.roi.outer_idx(2, "ravelled"), 5*4 + 2)

    def test_inverted(self):
        inv = np.ones((5,5), dtype=bool)
        inv[[2, 3, 4], [2, 2, 2]] = False
        self.assertTrue((self.roi.inverted.mask == inv).all())


class BoundaryTest_PurePython(unittest.TestCase):
    def setUp(self):
        self.get_boundary_mask = roi.get_boundary_mask_pure_python

    def test_name(self):
        self.assertEqual(self.get_boundary_mask.__name__, "get_boundary_mask_pure_python")

    def test_simple(self):
        m = np.zeros((5,5), dtype=np.bool)
        m[2,[2,3]] = True

        # Without diagonal neighbors.
        o = self.get_boundary_mask(m, withdiag=False)
        t = np.zeros((5,5), dtype=np.bool)
        t[2,[1,4]] = True
        t[1,[2,3]] = True
        t[3,[2,3]] = True
        self.assertTrue((o == t).all())

        # With diagonal neighbors.
        o = self.get_boundary_mask(m, withdiag=True)
        t = np.zeros((5,5), dtype=np.bool)
        t[2,[1,4]] = True
        t[1,[1,2,3,4]] = True
        t[3,[1,2,3,4]] = True
        self.assertTrue((o == t).all())

    def test_width(self):
        m = np.zeros((6,6), dtype=np.bool)
        m[2,[2,3]] = True

        # Without diagonal neighbors.
        o = self.get_boundary_mask(m, withdiag=False, width=2)
        t = np.zeros_like(m)
        t[2,[0,1,4,5]] = True
        t[1,[1,2,3,4]] = True
        t[3,[1,2,3,4]] = True
        t[0,[2,3]] = True
        t[4,[2,3]] = True
        self.assertTrue((o == t).all())

        # Without diagonal neighbors.
        o = self.get_boundary_mask(m, withdiag=True, width=2)
        t = np.zeros_like(m)
        t[2,[0,1,4,5]] = True
        t[1] = True
        t[3] = True
        t[0] = True
        t[4] = True
        self.assertTrue((o == t).all())

    def test_box(self):
        rect = roi.RectangleROI((20, 20), 2,4)
        t = roi.BoxROI(rect.shape, 2, 4, 4, 6)

        rect.move(6, 7)
        t.move(6, 7)
        b = self.get_boundary_mask(rect.mask, withdiag=True)
        self.assertTrue((b == t.mask).all())

        rect.move(6.5, 7.8)
        t.move(6.5, 7.8)
        b = self.get_boundary_mask(rect.mask, withdiag=True)
        self.assertTrue((b == t.mask).all())

        rect.move(0, 0)
        t.move(0, 0)
        b = self.get_boundary_mask(rect.mask, withdiag=True)
        self.assertTrue((b == t.mask).all())

    def test_where(self):
        """Test that only the arguments necessary are used."""
        y, x = 5, 5

        m = np.zeros((10, 10), dtype=np.bool)
        m[y, x] = True

        expected_wodiag = np.zeros_like(m)
        expected_wodiag[[y, y, y-1, y+1], [x-1, x+1, x, x]] = True

        # Test with where and out 
        out = self.get_boundary_mask(m, withdiag=False, where=([y], [x]))
        self.assertTrue((expected_wodiag == out).all())

        # Test with spot and out
        out = np.empty_like(m)
        self.get_boundary_mask(m, withdiag=False, out=out, where=([y], [x]))
        self.assertTrue((expected_wodiag == out).all())
        self.assertTrue((expected_wodiag == out).all())

    def test_boundary_of_diag_pattern_with_diag(self):
        m = np.zeros((10, 10), dtype=np.bool)
        m[3,2] = m[4,3] = True

        expected_w_diag = np.zeros_like(m)
        expected_w_diag[ [2,2,2,3,3,3,4,4,4,5,5,5],
                         [1,2,3,1,3,4,1,2,4,2,3,4] ] = True

        out = self.get_boundary_mask(m, withdiag=True)
        self.assertTrue((expected_w_diag == out).all())

    def test_boundary_of_diag_pattern_without_diag(self):
        m = np.zeros((10, 10), dtype=np.bool)
        m[3,2] = m[4,3] = True

        expected_wo_diag = np.zeros_like(m)
        expected_wo_diag[ [2,3,3,4,4,5],
                          [2,1,3,2,4,3] ] = True

        out = self.get_boundary_mask(m, withdiag=False)
        self.assertTrue((expected_wo_diag == out).all())


if roi.weave is not None:
    class BoundaryTest_Weave(BoundaryTest_PurePython):
        def setUp(self):
            self.get_boundary_mask = roi.get_boundary_mask_weave

        def test_name(self):
            self.assertEqual(self.get_boundary_mask.__name__, "get_boundary_mask_weave")


class HelperTest(unittest.TestCase):
    def test_py2_round(self):
        f = roi.round_afz

        self.assertIsInstance(f(1.0), int)
        self.assertEqual(1, f(1.49))
        self.assertEqual(2, f(1.5))
        self.assertEqual(2, f(1.7))

        self.assertEqual(2, f(2.4))
        self.assertEqual(3, f(2.5))
        self.assertEqual(3, f(2.7))

        self.assertEqual(3, f(3.4))
        self.assertEqual(4, f(3.5))
        self.assertEqual(4, f(3.7))

        self.assertEqual(0, f(-0.4))
        self.assertEqual(-1, f(-0.5))
        self.assertEqual(-1, f(-1.4))
        self.assertEqual(-2, f(-1.5))
        self.assertEqual(-2, f(-2.4))
        self.assertEqual(-3, f(-2.5))


if __name__ == "__main__":
    unittest.main()

