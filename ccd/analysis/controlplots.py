import numpy as np
from itertools import cycle
# Python{2,3} compatibility 
try:
    from itertools import izip as zip
except:
    pass

from matplotlib import pyplot as plt

def profiles(frames, axes=None, corr_offset=False):
    if axes is None:
        fig, axes = plt.subplots(2, sharey="all", figsize=(8,9))
        fig.subplots_adjust(left=0.16, right=0.96, hspace=0.25)
    # the frame is averaged perpendicular to the axis (0,1)
    data_0 = np.empty((len(frames), frames[0].shape[0]))
    data_1 = np.empty((len(frames), frames[0].shape[1]))

    for i, f in enumerate(frames):
        m0 = f.mean(axis=1)
        m1 = f.mean(axis=0)
        data_0[i] = m0
        data_1[i] = m1

    if corr_offset:
        data_0 -= np.mean(data_0, axis=0)
        data_1 -= np.mean(data_1, axis=0)

    colors = cycle("rgbm")
    linestyles = cycle(("-", "--", ":"))

    for i, c, ls in zip(xrange(len(frames)), colors, linestyles):
        axes[0].plot(data_0[i], color=c, linestyle=ls)
        axes[1].plot(data_1[i], color=c, linestyle=ls)

    axes[0].set_xlim(xmin=-10)
    axes[1].set_xlim(xmin=-10)

    axes[0].set_xlabel("row index $x_0$")
    axes[1].set_xlabel("column index $x_1$")
    if corr_offset:
        axes[0].set_ylabel("offset corr. column mean\n$m_{x_1}(v_f(x_0,x_1))-m_f(m_{x_1}(v_f(x_0,x_1)))$")
        axes[1].set_ylabel("offset corr. column mean\n$m_{x_0}(v_f(x_0,x_1))-m_f(m_{x_0}(v_f(x_0,x_1)))$")
    else:
        axes[0].set_ylabel("column mean\n$m_{x_1}(v_f(x_0,x_1))$")
        axes[1].set_ylabel("column mean\n$m_{x_0}(v_f(x_0,x_1))$")
    return axes
