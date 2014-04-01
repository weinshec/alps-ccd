import os.path
import numpy as np

from info import FrameInfo

class Frame(np.ndarray):
    """CCD frame container stores the pixel data in a numpy.array and meta
    data in the `info` dict-like attribute.

    Use `load_file` to load an image file.

    Index layout: [row, column] = [y, x]
    """

    # this list is filled before the specific load_* classmethods
    readable_extensions = []

    # see
    # <http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array>
    def __new__(cls, data, infoargs=None):
        obj = np.asarray(data).view(type=cls)
        if obj.ndim != 2:
            raise ValueError("%s can only hold 2D data. Use FrameSet for higher-dim arrays." % (cls))
        if infoargs is not None:
            # Copy arguments to circumvene sideeffects.
            obj.info = FrameInfo(**infoargs)
        else:
            obj.info = FrameInfo()

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "info", FrameInfo())

    ## Return only the plain array after operations to not confuse the info
    ## entries.
    # def __array_wrap__(self, arr, context=None):
        # """Called after some array operations (e.g. mean)."""
        # # if arr.shape == self.shape:
            # # return np.ndarray.__array_wrap__(self, arr, context)
        # return arr

    def __array_prepare__(self, arr, context=None):
        """Called before ufuncs."""
        return arr.view(np.ndarray)

    ## The arg... functions do not call __array_wrap__ or __array_prepare__
    ## and return objects of type `Frame`.
    def argmax(self, axis=None, out=None):
        return self.view(np.ndarray).argmax(axis, out)

    def argmin(self, axis=None, out=None):
        return self.view(np.ndarray).argmin(axis, out)

    def argsort(self, axis=-1, kind="quicksort", order=None):
        return self.view(np.ndarray).argsort(axis, kind, order)

    @classmethod
    def load_file(cls, path, info_extract=None, fileobj=None):
        """Load a file into a Frame or FrameSet.

        Supported file types: fits, spe, png.

        Parameters
        ----------
        path : str
            The path of the file.
        info_extract : callable, optional
            A callable to get informations from path for file types that do
            not support meta data (png at the moment). Must return a
            `FrameInfo` instance. Will be applied to the absolute path to
            get information from the directory names as well. See
            `InfoExtractions` object for possible candidates.
        """
        ext = os.path.splitext(path)[1].lower()

        if fileobj is None:
            fileobj = open(path, "rb")

        if ext == ".spe":
            return cls.load_spe(fileobj)

        elif ext in (".fits", ".fit"):
            return cls.load_fits(fileobj)

        elif ext == ".png":
            fi = FrameInfo()
            if info_extract is not None:
                ap = os.path.abspath(path)
                fi.update(info_extract(ap))
            return cls.load_png(fileobj , fi)

        else:
            raise ValueError("Unsupported file type: %s!" % ext)

    readable_extensions.extend([".fit", ".fits"])
    @classmethod
    def load_fits(cls, fileobj):
        import pyfits
        hdulist = pyfits.open(fileobj, mode="readonly")
        if len(hdulist) > 1:
            raise ValueError("More than 1 HDU in %s" % (fileobj,))
        phdu = hdulist[0]
        fi = FrameInfo.from_fits_header(phdu.header)
        fi["path"] = fileobj.name
        return cls(phdu.data, fi)

    readable_extensions.append(".spe")
    @classmethod
    def load_spe(cls, fileobj):
        from spe import SPEReader
        reader = SPEReader(file=fileobj)
        fi = FrameInfo()
        fi["temperature"] = reader.temp + 273.16 # convert from deg.Celsius to Kelvin
        fi["exposure"] = reader.exposure
        fi["datetime"] = reader.datetime
        fi["ro_mode"] = reader.readoutrate
        fi["camera"] = "PIXIS"
        fi["path"] = fileobj.name
        if reader.num_frames == 1:
            return cls(reader.data[0], fi)
        else:
            raise ValueError("multiple-frame SPE files are not supported by Frame")

    readable_extensions.append(".png")
    @classmethod
    def load_png(cls, fileobj, info):
        import png
        reader = png.Reader(file=fileobj)
        w, h, pngdata, meta = reader.asDirect()

        if not meta["greyscale"]:
            raise ValueError("Only grey scale PNG files supported!")

        png2uint = {8 : np.uint8, 16 : np.uint16, 32 : np.uint32}
        dtype = png2uint[meta["bitdepth"]]

        data = np.empty((reader.height, reader.width), dtype=dtype)
        for i, row in enumerate(pngdata):
            data[i] = np.asarray(row, dtype=dtype)

        info["camera"] = "XEVA"
        info["path"] = fileobj.name

        return cls(data, info)

    def write_to_fits(self, path, overwrite=False):
        """Write Frame data to FITS file.

        Parameter
        ---------
        path : str
            The file path.
        clobber : bool, optional
            Overwrite existing files. (default: False, existing files are
            not overwritten)
        """
        import pyfits
        hdu = pyfits.PrimaryHDU(self)
        self.info.update_fits_header(hdu.header)
        hdu.writeto(path, clobber=overwrite)
