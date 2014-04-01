#!/usr/bin/env python

import os.path
from copy import copy, deepcopy
import numpy as np
from scipy import stats
from frame import Frame
import info


class FrameSet(object):
    """Container for a set of frames taken under the same conditions
    (e.g. camera, temperature, exposure time)."""
    def __init__(self, frames=None):
        self._frames = None
        self.info = None
        # Default behaviour for self.get_frame(...)
        self.ret_plain_frame = False

        if frames is not None:
            self.add_frames(*frames)


    def write_to_fits(self, f, overwrite=False):
        """Write to open file-object `f`."""
        if hasattr(self.frames, "mask"):
            raise NotImplementedError("masked frames cannot be saved as FITS")

        import pyfits
        phdu = pyfits.PrimaryHDU()
        self.info.update_fits_header(phdu.header)
        hdus = [phdu]
        for i in xrange(self.num_frames):
            imhdu = pyfits.ImageHDU(data=self.get_frame(i, retplain=True))
            self.info.update_fits_header(imhdu.header, i)
            hdus.append(imhdu)
        hdus = pyfits.HDUList(hdus)
        hdus.writeto(f, clobber=overwrite)


    @classmethod
    def from_dir(cls, path, pattern="*", info_extract=None):
        """Load images from one directory.

        Only images that match the glob-pattern `pattern` are included.
        """
        from os.path import join
        if pattern:
            from glob import glob1
            names = (join(path, t) for t in glob1(path, pattern))
        else:
            from os import listdir
            names = (join(path, t) for t in listdir(path))
        names = sorted(names)
        frames = [Frame.load_file(n, info_extract) for n in names]
        return cls(frames)


    @classmethod
    def load_file(cls, path):
        """Load a frame set from a tar archive or FITS or SPE file."""
        ext = os.path.splitext(path)[-1].lower()
        if ext in (".tar", ".gz", ".bz2", ".tgz"):
            import tarfile
            tf = tarfile.open(path)

            # filter valid tar-file contents by file extensions supported by
            # Frame.load_file
            def ext_filter(tarinfo):
                t = os.path.splitext(tarinfo.name)[-1].lower()
                return t in Frame.readable_extensions

            fs = cls()
            tar_iext = info.TarFile(os.path.abspath(path))
            fs.info = tar_iext.setinfo

            names = [ti.name for ti in filter(ext_filter, tf.getmembers())]
            files = [tf.extractfile(c) for c in names]
            # frames = [Frame.load_file(n, fileobj=f) for (n, f) in zip(names, files)]
            frames = []
            for n, f in zip(names, files):
                f = Frame.load_file(n, fileobj=f)
                tar_iext.filepath = n
                f.info.update(tar_iext.frameinfo)
                fs.add_frames(f)

            return fs

        elif ext == ".spe":
            from spe import SPEReader
            reader = SPEReader(path=path)
            if reader.num_frames == 1:
                raise ValueError("single-frame SPE files are not supported by FrameSet")

            fi = info.FrameInfo()
            fi["temperature"] = reader.temp + 273.16 # convert from deg.Celsius to Kelvin
            fi["exposure"] = reader.exposure
            fi["datetime"] = reader.datetime
            fi["ro_mode"] = reader.readoutrate
            fi["camera"] = "PIXIS"
            frames = []
            for i in xrange(reader.num_frames):
                fi["comment"] = "%s Frame #%d" % (os.path.basename(path), i)
                fi["path"] = "%s%sframe %d" % (path, info.FrameSetInfo.framenumsep, i)
                frames.append(Frame(reader.data[i], fi))
            new = cls(frames)

            try:
                comment = info.tokenize_basename(path)["comment"].strip()
                if comment.startswith("PIXIS"):
                    comment = comment[5:].strip()
                new.info["set_comment"] = comment
            except info.InfoError:
                pass

            return new

        elif ext in (".fit", ".fits"):
            import pyfits
            hdulist = pyfits.open(path)
            if len(hdulist) < 3:
                raise ValueError("more than 1 Data HDU needed")

            ph = hdulist[0].header
            def mkframe(hdu, num):
                fi = info.FrameInfo.from_fits_header(hdu.header)
                fi.update_from_fits_header(ph)
                fi["path"] = "%s%sframe %d" % (path, info.FrameSetInfo.framenumsep, i)
                return Frame(hdu.data, fi)

            frames = [mkframe(hdu, num) for (num, hdu) in enumerate(hdulist[1:])]
            new = cls(frames)
            return new

        else:
            raise NotImplementedError("Loading frame sets from files of type %s is not implemented yet." % ext)


    def zoom_in(self, *args):
        """Return a frame set for the given pixel ranges.

        Accepted calling signatures::
            zoom_in(pixel_range)
            zoom_in(xmin, ymin, xmax, ymax)

        `pixel_range` must be an object with `xmin`, `ymin`, `xmax`, `ymax`
        attributes.
        """
        if len(args) == 1:
            arg = args[0]
            if arg is None:
                return self
            xmin = arg.xmin
            xmax = arg.xmax
            ymin = arg.ymin
            ymax = arg.ymax
        elif len(args) == 4:
            xmin, ymin, xmax, ymax = args
        else:
            # mimic standard python exception
            raise TypeError("zoom_in() takes exactly 2 or 5 arguments (%i given)" % len(args))

        new = self.__class__()
        new.info = deepcopy(self.info)
        new._frames = self._frames[:, ymin:ymax, xmin:xmax]
        return new

    def subset(self, indices, comment=""):
        """Return a frame set containing only the frames `indices`."""
        # copy the info data
        info = copy(self.info)
        if comment:
            if info.has_key("set_comment"):
                info["set_comment"] += " " + comment
            else:
                info["set_comment"] = comment

        def copy_lists(key):
            info[key] = [self.info[key][i] for i in indices]
        copy_lists("comments")
        copy_lists("datetime")
        copy_lists("paths")
        frames = self.frames[indices]

        new = self.__class__()
        new.info = info
        new._frames = frames
        return new

    def _update_info(self, frame):
        test_keys = ("camera", "temperature", "exposure", "ro_mode")
        # simple short-cut
        # fi = frame.info
        fi = getattr(frame, "info", dict())

        if self.info is None:
            self.info = info.FrameSetInfo()
            for key in test_keys:
                if fi.has_key(key):
                    self.info[key] = fi[key]

            self.info["comments"] = [fi.get("comment", "")]
            self.info["datetime"] = [fi.get("datetime")]
            self.info["paths"] = [fi.get("path", "")]

        else:
            for key in test_keys:
                if fi.has_key(key) and fi[key] != self.info[key]:
                    raise ValueError("Frame %s differs from set: %s vs. %s" % (key, fi[key], self.info[key]))
            self.info["comments"].append(fi.get("comment", ""))
            self.info["datetime"].append(fi.get("datetime"))
            self.info["paths"].append(fi.get("path", ""))


    def add_frames(self, *frames):
        """Add all arguments to the frame list."""
        if len(frames) == 0:
            raise ValueError("Frames required!")


        if self._frames is None:
            offset = 0
            frameshape = frames[0].shape

            shape = (len(frames), frameshape[0], frameshape[1])
            dtype = frames[0].dtype

            if any(hasattr(f, "mask") for f in frames):
                self._frames = np.ma.empty(shape,
                                           # `mask=...` is not necesarry:
                                           # `mask`s are silently created by
                                           # masked_array.
                                           # mask=np.zeros(shape, dtype=bool),
                                           dtype=dtype)
            else:
                self._frames = np.empty(shape, dtype=dtype)

        else:
            # copy old data
            offset = self.num_frames
            frameshape = self.frame_shape

            shape = (offset + len(frames), frameshape[0], frameshape[1])
            dtype = self.frames.dtype
            if hasattr(self._frames, "mask") or any(hasattr(f, "mask") for f in frames):
                new = np.ma.empty(shape,
                                  # mask=np.zeros(shape, dtype=bool),
                                  dtype=dtype)
            else:
                new = np.empty(shape, dtype=dtype)
            new[:offset] = self._frames
            self._frames = new

        for i, frame in enumerate(frames):
            if self._frames.dtype != frame.dtype:
                raise ValueError("Frame has incompatible data type: %s vs. %s!" % (self._frames.dtype, frame.dtype))
            if frameshape != frame.shape:
                raise ValueError("Frame has incompatible shape: %s vs. %s!" % (frameshape, frame.shape))
            self._update_info(frame)

            self._frames[i+offset] = frame


    @property
    def num_frames(self):
        if self._frames is not None:
            return self._frames.shape[0]
        return 0


    @property
    def frame_shape(self):
        return self._frames.shape[1:]


    @property
    def frames(self):
        return self._frames


    def get_frame(self, i, retplain=None):
        """Retrieve a stored frame. No copy is performed!

        Parameters
        ----------
        i : int
            Return the `i`th index.
        retplain : bool, optional
            If `True` return a plain `np.ndarray` frame, if `False` return a
            `Frame` object with appropriate `info` dictionary.
            If `None` (default) use the value of `self.ret_plain_frame`.

            If masked data is stored, the returned frame is always a masked frame.
        """
        if retplain is None:
            retplain = self.ret_plain_frame

        if retplain or hasattr(self._frames, "mask"):
            return self._frames[i]
        else:
            copy_keys = ("temperature", "exposure", "gain", "ro_mode", "camera")
            fi = info.FrameInfo((k, self.info[k]) for k in copy_keys if self.info.has_key(k))
            fi["datetime"] = self.info["datetime"][i]
            fi["comment"] = self.info["comments"][i]
            fi["path"] = self.info["paths"][i]
            return Frame(self._frames[i], fi)


    def mean_frame(self, retuncert=False, confidence=0.68, retframe=False):
        """Calculate the mean of each pixel, optionally the corresponding uncertainty.

        The uncertainty is calculated assuming the pixel-values are Gaussian
        distributed, thus their mean values are Student-t distributed. The
        uncertainty for a pixel is then calculated from the ISF of the
        t-distribution `t.isf`, the number of samples `n_meas` and the
        empirical standard deviation of the pixel values, `stdev`::

            ndf = n_meas - 1
            a_half = 0.5 * ( 1.0 - confidence)
            uncert = t.isf(a_half, ndf) * stdev / sqrt(n_meas)

        Parameters
        ----------
        retuncert : bool, optional
            Calculate the uncertainties. Default is not to calculate these.
        confidence : float, optional
            The confidence level for uncertainties. I.e. the _true_ pixel
            mean is with probability `confidence` within::

                mean - uncert < true < mean + uncert

        Returns
        -------
        mean : 2D array
            The mean pixel values.
        uncert : 2D array, optional
            The uncertainty of the mean values with `confidence` overlapping
            probability.
        """
        mean = self._frames.mean(0)
        if not retuncert:
            if retframe:
                # copy info attributes
                info = dict( (k, self.info[k])
                                for k in ["exposure", "camera", "ro_mode",
                                          "temperature"]
                                if self.info.get(k) )
                try:
                    path = self.info["path"]
                    info["comment"] = "mean frame of %s" % path
                    info["path"] = path
                except KeyError:
                    info["comment"] = "mean frame"
                mean = Frame(mean, infoargs=info)
            return mean
        elif retframe:
            raise ValueError("retuncert and retframe cannot be True at the same time!")

        n_meas = self.num_frames
        ndf = n_meas - 1
        # Probability in the upper-tail, that we don't cover with our band.
        a_half = 0.5 * (1.0 - confidence)

        std = self.stddev_frame(ddof=1)

        # Calculate the lower and upper confidence quantilles.
        quant = stats.t.isf(a_half, ndf)

        uncert = std * quant / np.sqrt(n_meas)

        return mean, uncert


    def stddev_frame(self, ddof=1, **kwargs):
        """Calculate the empirical standard deviation in each pixel.

        Parameters
        ----------
        ddof : int, optional
            The ddof parameter for `np.std`.
            For an unbiased estimator: `ddof=1` (default),
            for a maximum likelihood estimator: `ddof=0`.
        """
        return self._frames.std(0, ddof=ddof, **kwargs)


    def var_frame(self, retuncert=False, confidence=0.68, ddof=1, **kwargs):
        """Calculate the variance estimate in every pixel and optionally the
        associated uncertainties.

        The uncertainties are calculated based on the assumption that the
        variance is chi^2 distributed with :math:`N_{dof} = num_frames - 1`.
        The lower and upper chi2 quantilles are calculated and a
        symmetricized error estimate is returned.

        Parameters
        ----------
        ddof : int, optional
            The ddof parameter for `np.std`.
            For a unbiased estimator: `ddof=1` (default),
            for a maximum likelihood estimator: `ddof=0`.
        retuncert : bool, optional
            Return the estimate uncertainty is `True`. By default, the
            uncertainty is not returned.
        confidence : float, optional
            The confidence of the uncertainty estimation.

        Returns
        -------
        var : 2D array
            The estimated pixel variances.
        uncert : 2D array, optional
            The uncertainty of the estimated variances.
        """
        var = self._frames.var(0, ddof=ddof, **kwargs)
        if not retuncert:
            return var

        ndf = self.num_frames - 1
        a_half = 0.5 * (1.0 - confidence)

        # Calculate the lower and upper confidence quantilles.
        chi2 = stats.chi2(ndf)
        chi2_min = chi2.ppf(a_half)
        chi2_max = chi2.isf(a_half)

        uncert = 0.5 * ndf * var * (1.0/chi2_min - 1.0/chi2_max)

        return var, uncert


    def pixel_correlation_rms_frame(self, prnt=True):
        pixcorr = np.zeros(self.frame_shape)
        nx, ny = pixcorr.shape
        # means = self.mean_frame()
        # stddevs = self.stddev_frame()
        # a simple short-cut
        frames = self._frames

        def calc_pix_corr(ix, iy):
            val = 0.0
            # the values of the central pixel
            t_central = frames[:, ix, iy]
            for dx in xrange(-1, 2):
                for dy in xrange(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    t_neighbour = frames[:, ix+dx, iy+dy]
                    cor = np.corrcoef(t_central, t_neighbour)**2
                    assert cor.shape == (2, 2)
                    val += cor[0,1]**2

            return np.sqrt(val/8.0)

        for ix in xrange(1, nx-1):
            for iy in xrange(1, ny-1):
                pixcorr[ix, iy] = calc_pix_corr(ix, iy)
            if prnt:
                print "Finished column", ix, "of", nx-2
        return pixcorr
