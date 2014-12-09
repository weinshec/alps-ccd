from itertools import product
import numpy as np
from scipy import stats

from .. import roi

__all__ = ['IterativeFlagger', 'flag_per_row', 'flag_in_neighborhood', 'DirectNeighborFlagged']

# Use ravel_multi_index from numpy, if available, or define a replacement,
# if not available.
if hasattr(np, "ravel_multi_index"):
    ravel_multi_index = np.ravel_multi_index
else:
    def ravel_multi_index(mi, shape):
        if len(shape) != 2:
            raise ValueError("Only 2-dim arrays are allowed")
        if len(mi) != len(shape):
            raise ValueError("shape mismatch")
        return shape[0] * mi[0] + mi[1]



class IterativeFlagger(object):
    """Flag entries of `data` recursively until all remaining valid data is
    below `mean(valid data) + cut_s * std(valid data)`.
    """
    def __init__(self, data, cut_s, valid=None):
        self.orig_shape = data.shape
        if data.ndim > 1:
            self.data = data.ravel()
        else:
            self.data = data
        self.cut_s = cut_s
        self.idx = np.arange(self.data.size)
        if valid is None:
            self._valid = np.ones(self.data.shape, dtype=np.bool_)
        else:
            if valid.shape != self.orig_shape:
                raise ValueError("Shape of data and mask differ!")
            self._valid = valid.ravel()

    @property
    def valid(self):
        return self._valid.reshape(self.orig_shape)

    @property
    def flagged(self):
        return np.logical_not(self.valid)

    def __next__(self):
        # whole-data index of the maximal value of remaining valid data
        i_max = self.idx[self._valid][self.data[self._valid].argmax()]
        self._valid[i_max] = False
        m = self.data[self._valid].mean()
        s = self.data[self._valid].std(ddof=1)
        if self.data[i_max] >= m + self.cut_s * s:
            return i_max
        self._valid[i_max] = True
        raise StopIteration

    # Python2 compatibility
    next = __next__

    def __iter__(self):
        return self

    def flag(self):
        for i_max in self:
            pass


def flag_per_row(data, loss_ratio=None):
    """Flag outliers in data by examining the data row-wise with `IterativeFlagger`."""
    if data.ndim != 2:
        raise ValueError("data must be 2-dim")
    if loss_ratio is None:
        loss_ratio = 1.0 / data.shape[1]
    cut_s = stats.norm.isf(loss_ratio)
    wf_mask = np.empty(data.shape, dtype=np.bool_)

    for i, row in enumerate(data):
        flagger = IterativeFlagger(row, cut_s)
        flagger.flag()
        wf_mask[i] = flagger.flagged
    return wf_mask


class DirectNeighborFlagged(Exception):
    pass


def flag_in_neighborhood(data, sig_px, nh_size, nhff_size=0, loss_ratio=None):
    """
    Parameters
    ----------
    data : 2-dim ndarray
    sig_px : pair of ints
        The (axis0, axis1) (i.e. (y, x)) position of the signal pixel
    nh_size : int or pair of ints
        If an `int`, the size of the neighborhood, if a pair of `int`s, the
        (axis0, axis1) size of the neighborhood. Values must be odd.
    nhff_size : int, optional

    Returns
    -------
    flagged : 2-dim bool ndarray
        The mask with flagged pixels set to `True`.
    """
    if isinstance(nh_size, int):
        nh_size = (nh_size, nh_size)
    if nh_size[0] % 2 == nh_size[1] % 2 == 0:
        raise ValueError("Neighborhood size(s) must be odd")
    nh_roi = roi.BoxROI(data.shape, 1, nh_size[0], 1, nh_size[1])
    nh_roi.move(*sig_px)
    nh_data = data[nh_roi.mask]

    if loss_ratio is None:
        loss_ratio = 1.0 / nh_data.size
    cut_s = stats.norm.isf(loss_ratio)
    itf = IterativeFlagger(nh_data, cut_s)
    itf.flag()
    Y_flagged = []
    X_flagged = []
    for i_inner in np.nonzero(itf.flagged):
        i_wf = nh_roi.outer_idx(i_inner)
        y, x = np.unravel_index(i_wf, data.shape)
        Y_flagged.append(y)
        X_flagged.append(x)
    flagged = np.zeros(data.shape, dtype=np.bool_)
    flagged[Y_flagged, X_flagged] = True

    if nhff_size > 0:
        sig_y, sig_x = sig_px
        for d0, d1 in product(range(-nhff_size, nhff_size + 1), range(-nhff_size, nhff_size + 1)):
            if d0 == 0 and d1 == 0:
                continue
            if flagged[sig_y + d0, sig_x + d1]:
                err = DirectNeighborFlagged("Neighbor (y=%d, x=%d) of signal pixel (%d, %d) was flagged." % (sig_y+d0, sig_x+d1, sig_y, sig_x))
                err.sig_y = sig_y
                err.sig_x = sig_x
                err.d0 = d0
                err.d1 = d1
                err.flagged = flagged
                raise err
    return flagged
