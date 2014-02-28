"""A collection of patters and ROIs.
"""
import logging
import numpy as np
from itertools import izip, product

try:
    from scipy import weave
except ImportError:
    logging.error("Failed to import scipy.weave"
                  "\n  get_boundary_mask may be slow!")
    weave = None



class ROI(object):
    fmt = "<{self.__class__.__name__} object of shape:{self.shape}>"
    def __init__(self, shape):
        if len(shape) != 2:
            raise ValueError("2-dim shape expected")
        self.shape = shape
        self._mask = None

    def __str__(self):
        return self.fmt.format(self=self)

    def _get_mask(self):
        raise NotImplementedError("Method '_get_mask' must be implemented in sub-classes.")

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self._get_mask()
        return self._mask

    @property
    def n_px(self):
        return self.mask.sum()

    @property
    def ravelled(self):
        return np.arange(self.shape[0]*self.shape[1])[self.mask.flatten()]

    @property
    def yx(self):
        y, x = np.mgrid[:self.shape[0], :self.shape[1]]
        m = self.mask
        return y[m], x[m]

    @property
    def xy(self):
        y, x = self.yx
        return x, y

    def inner_idx(self, idx, which="ravelled"):
        """Convert whole-frame indices to indices valid in the ROI.

        Whole-frame indices that are not in the ROI are silently discarded.

        Calculate indices indexing pixels in the ROI from indices indexing
        the whole frame.

        I.e. indices starting with 0 in the ROI.

        Parameters
        ----------
        idx : array_like of ints
            The indices to convert.
        which : string
            Format of `idx`. At the moment only "ravelled" is accepted.

        Returns
        -------
        inner : array of ints
            The corresponding indices of the ROI area.

        Example
        -------
        >>> roi = ROI(asarray([[0, 0, 0, 0],
        ...                    [0, 1, 1, 0],
        ...                    [0, 1, 0, 0],
        ...                    [0, 0, 0, 0]]), dtype=bool)
        >>> roi.inner_idx(6)
        array([1])
        >>> roi.inner_idx(4)
        array([], dtype=int32)
        """
        if which == "ravelled":
            m = np.zeros(self.shape, dtype=np.bool).ravel()
            m[idx] = True
            m = m[self.ravelled]
        else:
            raise NotImplementedError("which '{0}' not implemented".format(which))

        return np.arange(self.n_px)[m]

    def outer_idx(self, idx, which="ravelled"):
        """Convert inner indices to indices valid in the whole frame.
        """
        if which == "ravelled":
            return self.ravelled[idx]
        else:
            raise NotImplementedError("which '{0}' not implemented".format(which))

    @property
    def inverted(self):
        inv = ROI(self.shape)
        inv._mask = np.logical_not(self.mask)
        return inv


class MovableROI(ROI):
    fmt = "<{self.__class__.__name__} object of shape:{self.shape} at [x:{self.x} y:{self.y}]>"
    def __init__(self, shape):
        super(MovableROI, self).__init__(shape)
        self.x = 0
        self.y = 0
        self.move(shape[0]//2, shape[1]//2)

    def move(self, i0, i1, xfirst=False):
        """Move the ROI to a new position on the frame.

        Parameters
        ----------
        i0, i1 : float
            The new position.
        xfirst : bool, optional
            If `True` interprete arguments as `x, y = i0, i1`. By default,
            the ordering `y, x = i0, i1` is assumed.
        """
        if xfirst:
            self.x = i0
            self.y = i1
        else:
            self.x = i1
            self.y = i0

        self._mask = None
        return self


class RectangleROI(MovableROI):
    fmt = "<{self.__class__.__name__}[width:{self.width} height:{self.height}] object of shape:{self.shape} at [x:{self.x} y:{self.y}]>"
    def __init__(self, shape, width, height):
        super(RectangleROI, self).__init__(shape)
        self.width = width
        self.height = height

    def _get_mask(self):
        mask = np.zeros(self.shape, dtype=np.bool)
        hh = 0.5 * self.height
        ymin = max(0, int(round(self.y - hh)))
        ymax = min(self.shape[0], int(round(self.y + hh)))

        hw = 0.5 * self.width
        xmin = max(0, int(round(self.x - hw)))
        xmax = min(self.shape[1], int(round(self.x + hw)))

        mask[ymin:ymax, xmin:xmax] = True
        return mask


class BoxROI(MovableROI):
    def __init__(self, shape, inner_width, outer_width, inner_height, outer_height):
        super(BoxROI, self).__init__(shape)
        self.inner_width = inner_width
        self.outer_width = outer_width
        self.inner_height = inner_height
        self.outer_height = outer_height

    def _get_mask(self):
        mask = np.zeros(self.shape, dtype=np.bool)

        # Simple short-cut
        rint = lambda x: int(round(x))

        # Set outer rectangle True
        hh = 0.5 * self.outer_height
        ymin = max(0, rint(self.y - hh))
        ymax = min(self.shape[0], rint(self.y + hh))

        hw = 0.5 * self.outer_width
        xmin = max(0, rint(self.x - hw))
        xmax = min(self.shape[1], rint(self.x + hw))

        mask[ymin:ymax, xmin:xmax] = True

        # Overwrite inner rectangle with False
        hh = 0.5 * self.inner_height
        ymin = max(0, rint(self.y - hh))
        ymax = min(self.shape[0], rint(self.y + hh))

        hw = 0.5 * self.inner_width
        xmin = max(0, rint(self.x - hw))
        xmax = min(self.shape[1], rint(self.x + hw))

        mask[ymin:ymax, xmin:xmax] = False

        return mask


class SquareROI(RectangleROI):
    """Square-shaped ROI.

    To center the pattern on a pixel use::

        >>> pattern.move(x + 0.5, y + 0.5)

    Parameters
    ----------
    shape : pair of int
        The shape of the 2-dim data `(y-len, x-len)`.
    size : int
        Size of the square.
    """
    fmt = "<{self.__class__.__name__}[size:{self.width}] object of shape:{self.shape} at [x:{self.x} y:{self.y}]>"
    def __init__(self, shape, size):
        super(SquareROI, self).__init__(shape, size, size)


class CircleROI(MovableROI):
    fmt = "<{self.__class__.__name__}[radius:{self.radius}] object of shape:{self.shape} at [x:{self.x} y:{self.y}]>"
    def __init__(self, shape, radius):
        super(CircleROI, self).__init__(shape)
        self.radius = radius

    @property
    def dr2(self):
        dy, dx = np.indices(self.shape)
        dy -= self.y
        dx -= self.x
        return dy**2 + dx**2

    def _get_mask(self):
        return self.dr2 <= self.radius**2


class RingROI(CircleROI):
    fmt = "<{self.__class__.__name__}[inner:{self.r_in} outer:{self.r_out}] object of shape={self.shape} at [x:{self.x} y:{self.y}]>"
    def __init__(self, shape, r_in, r_out):
        super(CircleROI, self).__init__(shape)
        self.r_in = r_in
        self.r_out = r_out

    def _get_mask(self):
        dr2 = self.dr2
        return np.logical_and(self.r_in**2 <= dr2, dr2 < self.r_out**2)


class IrregularROI(ROI):
    def __init__(self, mask):
        super(IrregularROI, self).__init__(mask.shape)

        self._mask = mask


class SliceROI(ROI):
    """Construct a ROI from slices.

    Parameters
    ----------
    shape : (int, int)
        The shape of the target mask.
    sl0, sl1: slice or (int, int)
        The slices of the ROI.
    """
    def __init__(self, shape, sl0, sl1):
        super(SliceROI, self).__init__(shape)
        if isinstance(sl0, tuple):
            sl0 = slice(sl0[0], sl0[1])
        if isinstance(sl1, tuple):
            sl1 = slice(sl1[0], sl1[1])

        self.sl0 = sl0
        self.sl1 = sl1

    def _get_mask(self):
        mask = np.zeros(self.shape, dtype=np.bool)
        mask[self.sl0,self.sl1] = True
        return mask



# TODO: implement boundary width of more than one pixel

class BoundaryROI(ROI):
    fmt = "<{self.__class__.__name__}[withdiag={self.withdiag} width={self.width}] of {self._central_roi} object>"
    def __init__(self, central_roi, withdiag=True, width=1):
        super(BoundaryROI, self).__init__(central_roi.shape)
        self.withdiag = withdiag
        self._central_roi = central_roi
        self.width = width

    def _get_mask(self):
        return get_boundary_mask(self._central_roi.mask, self.withdiag, self.width)


# This is slower than the pure python implementation.
# from _get_boundary import get_boundary as _c_gbm

# def get_boundary_map(spot):
    # bounds = np.zeros(spot.shape, dtype=np.int)
    # _c_gbm(np.asarray(spot, dtype=np.int), bounds)
    # return np.asarray(bounds, dtype=np.bool)


def get_boundary_mask_pure_python(spot, withdiag=False, width=1, out=None, where=None):
    """Get the boundary of a single, connected spot.

    If the spot is not singular, the behaviour is untested.

    Parameters
    ----------
    spot : 2-dim bool array
    withdiag : bool, optional
        If `True` include diag. neighbors in the boundary map.
    width : int, optional
        The width of the neighborhood in pixels. For `width > 1` the result
        differs strongly depending the value of  `withdiag`.
    out : 2-dim array, optional
        If not `None` the result is prepared in `out` and a reference to
        `out` is returned.
    where : (y-indices, x-indices), optional
        If given, the lists of y- and x-indices of the spots pixels. If
        `None` the indices are computed using `np.where`, which can be
        time-consuming.
    """
    if out is None:
        bounds = np.empty_like(spot)
    else:
        if where is None and out.shape != spot.shape:
            raise ValueError("shape mismatch")
        bounds = out
    bounds.fill(False)

    t = range(-width, width+1)
    if withdiag:
        idx_deltas = list(product(t, t))
    else:
        idx_deltas = [(dx, dy) for dx in t for dy in t if (abs(dx) + abs(dy) <= width)]
    idx_deltas.remove((0, 0))

    n_y, n_x = bounds.shape
    if where is None:
        # TODO: np.where is inefficient for `spot`s with large shapes
        Y, X = np.where(spot)
    else:
        Y, X = where
    for y, x in izip(Y, X):
        for dy, dx in idx_deltas:
            t = y + dy
            s = x + dx
            # valid indices
            if t < 0 or t > n_y - 1 or s < 0 or s > n_x - 1:
                continue
            if not spot[t, s]:
                bounds[t, s] = True
    return bounds


def get_boundary_mask_weave(spot, withdiag=False, width=1, out=None, where=None):
    """Get the boundary of a single, connected spot.

    If the spot is not singular, the behaviour is untested.

    Parameters
    ----------
    spot : 2-dim bool array
    withdiag : bool, optional
        If `True` include diag. neighbors in the boundary map.
    width : int, optional
        The width of the neighborhood in pixels. For `width > 1` the result
        differs strongly depending the value of  `withdiag`.
    out : 2-dim array, optional
        If not `None` the result is prepared in `out` and a reference to
        `out` is returned.
    where : (y-indices, x-indices), optional
        If given, the lists of y- and x-indices of the spots pixels. If
        `None` the indices are computed using `np.where`, which can be
        time-consuming.
    """
    if out is None:
        bounds = np.empty_like(spot)
    else:
        if where is None and out.shape != spot.shape:
            raise ValueError("shape mismatch")
        bounds = out
    bounds.fill(False)

    n_y, n_x = bounds.shape
    if where is None:
        # TODO: np.where is inefficient for `spot`s with large shapes
        Y, X = np.where(spot)
    else:
        # NOTE: Convert to `ndarray`s to make the indexing in the C-code
        #       painless.
        Y = np.asarray(where[0])
        X = np.asarray(where[1])

    # some helpers for the weave code
    n_px = len(Y)
    withdiag = int(withdiag)
    code = r""" // C
    int d0, d1, i0, i1, w1;
    int j;


    for (j = 0; j < n_px; ++j) {
      i0 = Y(j);
      i1 = X(j);
      for (d0 = -width; d0 <= width; ++d0) {
        w1 = (withdiag) ? width : width - abs(d0);
        for (d1 = -w1; d1 <= w1; ++d1) {
          if (d0 == 0 && d1 == 0)
            continue;
          if ( i0 + d0 < 0 || i1 + d1 < 0 ||
               i0 + d0 >= n_y || i1 + d1 >= n_x )
            continue;

          if ( !spot(i0 + d0, i1 + d1)) {
            bounds(i0 + d0, i1 + d1) = 1;
          }
        }
      }
    }  
    """
    weave.inline(code, ["n_px", "Y", "X", "width", "withdiag", "n_y", "n_x", "spot", "bounds"],
                      type_converters = weave.converters.blitz)
    return bounds

if weave is None:
    get_boundary_mask = get_boundary_mask_pure_python
else:
    get_boundary_mask = get_boundary_mask_weave
