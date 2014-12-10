import numpy as np
import scipy.optimize

__all__ = ["fwhm", "GaussContExpo", "Gauss2D"]

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
        return [fwhm(data, ax) for ax in range(data.ndim)]

    axes = np.r_[0:axis, axis+1:data.ndim]
    d = np.apply_over_axes(np.mean, data, axes).flatten()
    imax = d.argmax()
    hmax = 0.5 * d[imax]
    i0 = np.where(d[:imax] <= hmax)[0][-1]
    i1 = np.where(d[imax:] <= hmax)[0][0] + imax
    return i0, i1


class GaussContExpo(object):
    """Fit a 2-dim Gaussian with exponential tail to `data`.

    Before fitting 2-dim data, one has to initialise the parameters of the
    fit-function. Parameters can be set using :meth:`update_params`, during
    object creation or with :meth:`initial_guess`.

    The current parameter values are accesible through :attr:`param_values`
    and :attr:`param_dict`.
    """
    param_names = ("A", "mu0", "mu1", "sigma0", "sigma1", "xi", "offset")
    def __init__(self, *args, **kwargs):
        self.param_values = [None] * len(self.param_names)
        if args or kwargs:
            self.update_params(*args, **kwargs)

    def update_params(self, *args, **kwargs):
        """Update fit-function parameters.

        This method accepts either the numerical values of *all* parameters as arguments::

            >>> fitter.update_params(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        or keyword arguments (not necessarily all parameters)::

            >>> fitter.update_params(mu0=2.0, mu1=3.0)

        """
        if args and kwargs:
            raise ValueError("Only *args or **kwargs accepted")
        if args:
            if len(args) != len(self.param_names):
                raise ValueError("Wrong number of parameters in *args")

            self.param_values = list(args)
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
        """Evaluate the fit function at position (X0, X1) using the current parameter values.

        Parameters
        ----------
        X0, X1 : float, ndarray
            The locations where the fit-funtion is evaluated. Using
            :func:`matplotlib.matshow` convention means `y, x = X0, X1`.
        """
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


class Gauss2D(GaussContExpo):
    """Fit a 2-dim to `data`.


    See
    ---
    :class:`GaussContExpo`
    """
    param_names = ("A", "mu0", "mu1", "sigma0", "sigma1", "theta", "offset")

    @staticmethod
    def func(X0_X1, amplitude, mu0, mu1, sigma0, sigma1, theta, offset):
        """Analytic definition of 2D gaussian distribution"""
        X0, X1 = X0_X1

        # See http://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        # and http://demonstrations.wolfram.com/IntuitiveParameterizationOfTheBivariateNormalDistribution/
        # We assume that array-index 0 corresponds to y and array-index 1 to
        # x.
        st2 = np.sin(theta)**2
        ct2 = np.cos(theta)**2
        sigma0_2 = sigma0**2
        sigma1_2 = sigma1**2
        a = 0.5 * (ct2 / sigma1_2 + st2 / sigma0_2)
        b = 0.25 * np.sin(2*theta) * (1 / sigma0_2 - 1 / sigma1_2)
        c = 0.5 * (st2 / sigma1_2 + ct2 / sigma0_2)
        g = offset + amplitude * np.exp(-(a * (X1-mu1)**2
                                          + c * (X0-mu0)**2
                                          + 2 * b * (X1-mu1) * (X0-mu0)
                                          ))
        return g

    def initial_guess(self, data):
        """Guess distribution parameters based on given dataset

        Parameters:
        -----------
        data : 2-dim ndarray
            data numpy array of dimension 2
        """
        mu0, mu1 = np.unravel_index(data.argmax(), data.shape)
        fwhm0 = fwhm(data[:,mu1] - data[:,mu1].min())[0]
        fwhm1 = fwhm(data[mu0] - data[mu0].min())[0]

        self.update_params(
                data.max() - data.min(),
                mu0, mu1,
                (fwhm0[1] - fwhm0[0]) / 2.35,
                (fwhm1[1] - fwhm1[0]) / 2.35,
                0.0,
                data.min()
                )

    def fit(self, data):
        """Perform least-squares fit of 2-dim gaussian to given data.

        Parameters
        ----------
        data : 2-dim ndarray
            The data to be fitted.
        """
        if None in self.param_values:
            raise ValueError("Not all parameter values initialized")

        n0, n1 = data.shape
        X0, X1 = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")

        X0 = X0.ravel()
        X1 = X1.ravel()
        data = data.ravel()
        # x = np.linspace(0, size[0], size[0])
        # y = np.linspace(0, size[1], size[1])
        # xy = np.meshgrid(x, y)

        popt, pcov = scipy.optimize.curve_fit(self.func, (X0, X1), data, p0=self.param_values)
        self.update_params(*popt)

    def evaluate(self, X0, X1, ignoreOffset=False):
        """Evaluate the 2D gaussian function at a given point

        Parameters
        ----------
        X0, X1 : float, ndarray
            The locations where the fit-funtion is evaluated. Using
            :func:`matplotlib.matshow` convention means `y, x = X0, X1`.

        Returns
        -------
        v : float, ndarray
            The values of the gaussian  at this position
        """
        if ignoreOffset:
            return self.func((X0, X1), *self.param_values) - self.param_dict['offset']
        else:
            return self.func((X0, X1), *self.param_values)

    def integral(self):
        """Calculate the integral under the spot ignoring the offset"""
        A, mu0, mu1, sigma0, sigma1, theta, offset = self.param_values

        return A * 2 * np.pi * sigma0 * sigma1
