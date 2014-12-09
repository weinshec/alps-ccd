from functools import wraps
import numpy as np

from ccd.io.frame import Frame


def safe_slice(a):
    """If possible convert `a` into a `slice` object.

    Returns
    -------
    ret : slice, other
        If possible, a slice object, or `a`.
    issclice : bool
        `True` if `ret` is a `slice`.
    """
    if a is None:
        return safe_slice((None, None))
    elif type(a) == slice:
        return a, True
    elif len(a) == 2:
        return slice(a[0], a[1]), True
    return a, False


class plain_offset(object):
    """Estimate the offset from averaging over all pixel in `offset_rows` and `offset_cols`.

    Parameters
    ----------
    offset_rows, offset_cols: valid ndarray index object, e.g. slice or None
        The rows and columns where the offset will be estimated. If `None`,
        all rows or columns are used for estimation, respectively.
    frame : 2dim ndarray, Frame, optional
        The frame to estimate from. If `None` (default) a callable object is
        returned.
    """
    def __new__(cls, offset_rows=None, offset_cols=None, frame=None):
        if frame is None:
            obj = super(plain_offset, cls).__new__(cls, offset_rows, offset_cols)
            return obj
        else:
            return cls(offset_rows, offset_cols)(frame)

    def __init__(self, offset_rows, offset_cols):
        self.offset_rows, isslice_rows = safe_slice(offset_rows)
        self.offset_cols, isslice_cols = safe_slice(offset_cols)
        self.comment = "row-wise offset estimated from averaged"
        if isslice_rows:
            self.comment += " rows [{0}:{1}]".format(
                    self.offset_rows.start if self.offset_rows.start else "",
                    self.offset_rows.stop if self.offset_rows.stop else "")
        else:
            self.comment += " rows {0}".format(self.offset_rows)
        if isslice_cols:
            self.comment += " columns [{0}:{1}]".format(
                    self.offset_cols.start if self.offset_cols.start else "",
                    self.offset_cols.stop if self.offset_cols.stop else "")
        else:
            self.comment += " columns {0}".format(self.offset_cols)

    def __call__(self, frame):
        return frame.view(np.ndarray)[self.offset_rows, self.offset_cols].mean()


class rowwise_offset(object):
    """Estimate the row-wise offset from averaging over a given range of
    columns.

    If this correction is performed the `plain_offset` correction is
    redundant.

    Parameters
    ----------
    offset_cols: valid ndarray index object, e.g. slice or None
        The columns where the per-row offset will be estimated.
    frame : 2dim ndarray, Frame, optional
        The frame to estimate from. If `None` (default) a callable object is
        returned.
    """
    def __new__(cls, offset_cols, frame=None):
        if frame is None:
            obj = super(rowwise_offset, cls).__new__(cls)
            return obj
        else:
            return cls(offset_cols)(frame)

    def __init__(self, offset_cols):
        self.offset_cols, isslice = safe_slice(offset_cols)
        if isslice:
            self.comment = "row-wise offset estimated from averaged columns [{0}:{1}]".format(
                    self.offset_cols.start if self.offset_cols.start else "",
                    self.offset_cols.stop if self.offset_cols.stop else "")
        else:
            self.comment = "row-wise offset estimated from averaged columns {0}".format(
                    self.offset_cols)

    def __call__(self, frame):
        return frame.view(np.ndarray)[:, self.offset_cols].mean(axis=1)[:, np.newaxis]


def _check_shape(f):
    """Decorator that checks that a method is passed a frame with correct shape."""
    @wraps(f)
    def new_f(self, frame, *args, **kwargs):
        if frame.shape != self.valid_shape:
            raise ValueError("Given frame has wrong shape: %r" % (frame.shape,))
        return f(self, frame, *args, **kwargs)
    return new_f


class FrameNormalizer(object):
    """A little helper to correct PIXIS frames using the overscan columns."""
    def __init__(self, valid_shape, data_rows, data_cols, offset_estimator, get_fpn=None):
        """
        Parameters
        ----------
        valid_shape : tuple
            (nrows, ncolumns)
        data_rows, data_cols:
            The rows and columns of the data region. `(a,b)` pairs are
            converted to slices. See :func:`safe_slice`.
        offset_estimator : callable
            Callable that returns offset frame, which is subtracted from
            data. See :meth:`correct_offset`.
        """
        self.valid_shape = valid_shape
        self.data_rows = safe_slice(data_rows)[0]
        self.data_cols = safe_slice(data_cols)[0]
        self.offset_estimator = offset_estimator
        self.get_fpn = get_fpn

    @_check_shape
    def correct_offset(self, frame):
        offset = self.offset_estimator(frame)
        r = frame.view(np.ndarray) - offset
        if hasattr(frame, "info"):
            info = frame.info.copy()
            cmt = "offset subtracted"
            cmt2 = getattr(self.offset_estimator, "comment")
            if cmt2:
                cmt += " ({0})".format(cmt2)
            if info.get("comment"):
                info["comment"] += "; " + cmt
            else:
                info["comment"] = cmt
            r = Frame(r, info)
        return r

    @_check_shape
    def data_region(self, frame):
        return frame[self.data_rows, self.data_cols]

    @_check_shape
    def subtract_fpn(self, frame, *args, **kwargs):
        fpn = self.get_fpn(frame, *args, **kwargs)
        r = frame.view(np.ndarray) - fpn.view(np.ndarray)
        if hasattr(frame, "info"):
            info = frame.info.copy()
            try:
                fpn_path = fpn.info["path"]
                cmt = "FPN('{0}') subtracted".format(fpn_path)
            except (AttributeError, KeyError):
                cmt = "FPN subtracted"

            if info.get("comment"):
                info["comment"] += "; " + cmt
            else:
                info["comment"] = cmt
            r = Frame(r, info)
        return r
