import numpy as np
import scipy.optimize

__all__ = ["GaussContExpo"]

def fwhm(data, axis=None):
    """Calculate the indices of the FWHM for the projections on the axis.

    Parameters
    ----------
    data : array_like
        n-dim data
    axis : int, optional
        The axis on which the projection is taken. If axis is `None` return
        a list of all FWHM index pairs.

    Returns
    -------
    idx : list of pairs of int
        The indices of the fwhm. If axis is specified a plain pair is
        returned.

    See
    ---
    For usage of `apply_over_axes` see:
    http://www.mail-archive.com/numpy-discussion@lists.sourceforge.net/msg03469.html
    """
    if axis is None:
        return [fwhm(data, ax) for ax in xrange(data.ndim)]

    axes = np.r_[0:axis, axis+1:data.ndim]
    d = np.apply_over_axes(np.mean, data, axes).flatten()
    imax = d.argmax()
    hmax = 0.5 * d[imax]
    i0 = np.where(d[:imax] <= hmax)[0][-1]
    i1 = np.where(d[imax:] <= hmax)[0][0] + imax
    return i0, i1


class GaussContExpo(object):
    """Fit a 2D Gaussian with exponential tail to `data`."""
    param_names = ("A", "mu0", "mu1", "sigma0", "sigma1", "xi", "offset")
    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            self.param_values = [None] * len(self.param_names)
        else:
            self.update_params(*args, **kwargs)

    def update_params(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Only *args or **kwargs accepted")
        if args:
            if len(args) != len(self.param_names):
                raise ValueError("Wrong number of parameters in *args")

            self.param_values = args
        else:
            for k, v in kwargs.items():
                i = self.param_names.index(k)
                self.param_values[i] = v

    @property
    def param_dict(self):
        return dict(zip(self.param_names, self.param_values))

    def initial_guess(self, data):
        mu0, mu1 = np.unravel_index(data.argmax(), data.shape)
        fwhm0 = fwhm(data[:,mu1] - data[:,mu1].min())[0]
        fwhm1 = fwhm(data[mu0] - data[mu0].min())[0]

        self.update_params(
                data.max() - data.min(),
                mu0, mu1,
                (fwhm0[1] - fwhm0[0]) / 2.35,
                (fwhm1[1] - fwhm1[0]) / 2.35,
                1.0,
                data.min()
                )

    def fit(self, data, n_fit):
        if None in self.param_values:
            raise ValueError("Not all parameter values initialized")

        X0 = np.random.randint(0, data.shape[0], n_fit)
        X1 = np.random.randint(0, data.shape[1], n_fit)

        D = data[X0, X1]
        def diff(p):
            self.update_params(*p)
            Z = self.evaluate(X0, X1)
            return np.sum((D - Z)**2)


        p_opt = scipy.optimize.fmin(diff, self.param_values)
        self.update_params(*p_opt)

    def evaluate(self, X0, X1):
        A, mu0, mu1, sigma0, sigma1, xi, offset = self.param_values

        out = np.empty(X0.shape, dtype=np.float)

        a = 1.0 * (X0 - mu0) / sigma0
        b = 1.0 * (X1 - mu1) / sigma1
        R2 = a**2 + b**2

        B = A * np.exp(0.5 * xi**2)

        idx = (R2 <= xi**2)
        out[idx] = A * np.exp(-0.5*R2[idx]) + offset

        idx = (R2 > xi**2)
        out[idx] = B * np.exp(-xi*np.sqrt(R2[idx])) + offset
        return out

    def integral(self):
        """Calculate the integral under the spot ignoring the offset."""
        A, mu0, mu1, sigma0, sigma1, xi, offset = self.param_values
        return 2.0 * np.pi * sigma0 * sigma1 * A * (1.0 + np.exp(-0.5 * xi**2) / xi**2)
