"""
Functions to visualize spotlists.
"""

from itertools import groupby

import numpy as np

from ...tools.plotting import axes_helper, kw_helper, share_axis, zmatshow
from ..info import mk_key_function

from spotlist import spot_mask, spot_integ


def spot_integ_hist(spotlists, ax=None, correctbgd=True, **histkw):
    """Show spot-height histograms.

    Spot-lists are combined for frames with the same exposure time from the
    same data-set (i.e. sub-directory).
    """
    ax = axes_helper(ax)

    histkw = kw_helper(dict(histtype="stepfilled", alpha=0.5), histkw)

    for (d, expos), sub_sls in groupby(spotlists, mk_key_function("dir", "exposure")):
        l = "dir: %s\nexpos: %f" % (d, expos)
        print l
        integs = []
        for sl in sub_sls:
            integs.append(spot_integ(sl, correctbgd))
        integs = np.concatenate(integs)
        ax.hist(integs, label=l, **histkw)


def spotlist_figure(spotlist):
    from matplotlib import pyplot
    f = pyplot.figure(figsize=(16,7))

    ax_spots = f.add_axes([0.1, 0.2, 0.4, 0.7])
    ax_cb_spots = f.add_axes([0.1, 0.1, 0.4, 0.08])

    ax_frame = f.add_axes([0.55, 0.2, 0.4, 0.7])
    ax_cb_frame = f.add_axes([0.55, 0.1, 0.4, 0.08])

    aimg = ax_spots.matshow(spot_mask(spotlist), cmap=pyplot.cm.gray_r)
    pyplot.colorbar(mappable=aimg, cax=ax_cb_spots, orientation="horizontal")

    aimg = zmatshow(spotlist.frame, ax=ax_frame, cmap=pyplot.cm.gist_ncar)
    pyplot.colorbar(mappable=aimg, cax=ax_cb_frame, orientation="horizontal")

    share_axis([ax_spots, ax_frame], sharex=True, sharey=True)
