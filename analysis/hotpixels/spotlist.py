"""
:class:`SpotList` is a container for lists of spots that can easily be
dumped and loaded with `pickle`. Additionally, this module contains some
functions that analyse the data of a spot list.
"""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from scipy import ndimage

__all__ = ['SpotList', 'spot_mask', 'group_pixels', 'spot_integral', 'filter_list', 'pred_npx_equals', 'pred_npx_gtr']

class SpotList(list):
    """Container for spots.

    A spot is a list of ravelled indices.

    Can be `dump`ed and `load`ed.
    """
    def __init__(self, shape, spots=None, **kwargs):
        self.shape = shape
        if spots is not None:
            super(SpotList, self).__init__(spots)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def dump(self, path):
        with open(path, "w") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a pickled spot list.

        Parameters
        ----------
        path : str
            The path of the pickle file.
        loadframe : bool, optional
            If `True` (default) load the attached frame. If `False` the
            frame is not loaded.
        """
        with open(path) as f:
            new = pickle.load(f)
        return new


def spot_mask(spotlist, shape=None):
    """Return a boolean mask-array with the pixels of the spots are `True`.

    Parameters
    ----------
    spotlist : list, SpotList
        A list of spots. Here a spot is simply a list of the ravelled pixel
        indices that belong to the spot.
    shape : tuple of int, optional
        The shape of the frame. If no specified `spotlist` should have a
        `frame` attribute, which is array-like.
    """
    if shape is None:
        shape = spotlist.shape
    if len(shape) != 2:
        raise ValueError("shape must be 2-dim")
    size = shape[0] * shape[1]
    m = np.zeros(size, dtype=np.bool)
    for spot in spotlist:
        m[spot] = True
    return m.reshape(shape)


def group_pixels(spotlist):
    """Group adjacent masked pixels into spots."""
    mask = spot_mask(spotlist)
    new = SpotList(spotlist.shape)
    for k in spotlist.__dict__:
        if not hasattr(new, k):
            setattr(new, k, getattr(spotlist, k))
    labeled, n_spots = ndimage.label(mask, structure=np.ones((3,3)))
    for i in xrange(n_spots):
        t = list(np.flatnonzero(i+1 == labeled))
        assert len(t) > 0
        new.append(t)
    return new


# def masked_frameset(spotlists):
    # """Return a `FrameSet` containing masked arrays."""
    # return FrameSet([np.ma.masked_array(sl.frame, mask=spot_mask(sl)) for sl in spotlists])


def spot_integral(spotlist, correctbgd=True):
    """Calculate the integral under each spot in `spotlist`.

    If `correctbgd == True` the integral under each spot is corrected for
    the background, which is estimated by the average of all pixels that are
    in no spot.
    """
    d_rav = spotlist.frame.ravel()
    integs = np.empty(len(spotlist))

    if correctbgd:
        no_spots = np.logical_not(spot_mask(spotlist))
        m = spotlist.frame[no_spots].mean()
        for i, spot in enumerate(spotlist):
            integs[i] = d_rav[spot].sum() - len(spot) * m
    else:
        for i, spot in enumerate(spotlist):
            integs[i] = d_rav[spot].sum()

    return integs


def filter_list(spotlist, predicate):
    """Filter a spot list with `predicate`.

    Parameters
    ----------
    spotlist : SpotList, list
    predicate : callable
        Spots with `predicate(spotlist, spot) == True` are added to the
        returned list.

        For possible predicates see :func:`pred_npx_equals` and
        :func:`pred_npx_gtreq`.

    Returns
    -------
    new_list : SpotList, list
        A list of spots from `spotlist` with all spots where
        `predicate(spot)` is `True`.
    """
    return SpotList(spotlist.shape,
                    spots=(spt for spt in spotlist if predicate(spotlist, spt)))


def pred_npx_equals(npx):
    def predicate_equals(spotlist, spt):
        return len(spt) == npx
    return predicate_equals


def pred_npx_gtr(npx):
    def predicate_gtr(spotlist, spt):
        return len(spt) > npx
    return predicate_gtr
