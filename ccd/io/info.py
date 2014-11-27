import os.path
import re
from operator import indexOf
import datetime

class InfoError(Exception):
    pass


def _convert_time2seconds(t):
    """Convert a time-string to a float of seconds."""
    if t.endswith("us"):
        return float(t[:-2]) * 1e-6
    if t.endswith("ms"):
        return float(t[:-2]) * 1e-3
    if t.endswith("secs"):
        return float(t[:-4])

    raise ValueError("Failed to parse time string '%s'" % t)


def _convert_temperature2kelvin(t):
    if t.endswith("K"):
        return float(t[:-1])
    if t.endswith("C"):
        return float(t[:-1]) + 273.16

    raise ValueError("Failed to parse temperature string '%s'" % t)



def get_date(part):
    t = part.split()[0]
    year, month, day = map(int, t.split("-"))
    return datetime.datetime(year, month, day)


def tokenize_basename(path):
    basename = os.path.basename(path)
    m = re.search(r"(?P<comment>.*) (?P<exposure>\S*[a-z]) (?P<temperature>\S*[KC])", basename)
    try:
        return m.groupdict()
    except Exception as err:
        raise InfoError("Failed to tokenize basename({0!r}): {1}".format(path, err))




class FrameInfo(dict):
    _allowed_keys = ("temperature", "exposure", "gain", "datetime",        "ro_mode", "camera", "comment", "path")
    _allowed_types = (float,        float,      str,    datetime.datetime, str,       str,      str,       str)

    def __init__(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v

    def __setitem__(self, key, new):
        try:
            idx = indexOf(self._allowed_keys, (key))
        except ValueError:
            raise ValueError("key '%s' not allowed" % key)

        t = self._allowed_types[idx]
        if type(new) != t:
            try:
                new = t(new)
            except StandardError as e:
                raise TypeError("Conversion to %s failed for %s: %s" % (t, key, e))

        dict.__setitem__(self, key, new)

    def set(self, key, new):
        self[key] = new

    @classmethod
    def from_fits_header(cls, header):
        new = cls()
        new.update_from_fits_header(header)
        return new

    def update_from_fits_header(self, header):
        # Parse my CCDOPS FITS header.
        if header.get("INSTRUME") == "SBIG ST-402":
            self["temperature"] = header["CCD-TEMP"]
            self["exposure"] = header["EXPTIME"]
            self["gain"] = header["EGAIN"]
            self["datetime"] = datetime.datetime.strptime(header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f")
            self["ro_mode"] = "normal"
            self["camera"] = "SBIG ST-402"
        # Parse my own FITS header.
        else:
            if header.has_key("TEMP"):
                self["temperature"] = header["TEMP"]

            if header.has_key("EXPOSURE"):
                self["exposure"] = header["EXPOSURE"]

            if header.has_key("GAIN"):
                self["gain"] = header["GAIN"]

            if header.has_key("DATETIME"):
                try:
                    t = datetime.datetime.strptime(header["DATETIME"], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    t = datetime.datetime.strptime(header["DATETIME"], "%Y-%m-%dT%H:%M:%S.%f")
                self["datetime"] = t

            if header.has_key("ROMODE"):
                self["ro_mode"] = header["ROMODE"]

            if header.has_key("CAMERA"):
                self["camera"] = header["CAMERA"]

            # Use non-special key COMMNT
            if header.has_key("COMMNT"):
                self["comment"] = header["COMMNT"]
            elif header.has_key("COMMENT"):
                self["comment"] = header["COMMENT"]

    def update_fits_header(self, header):
        if self.has_key("temperature"):
            header.update("TEMP", self["temperature"], "Detector temperature")

        if self.has_key("exposure"):
            header.update("EXPOSURE", self["exposure"], "Exposure time in secs")

        if self.has_key("gain"):
            header.update("GAIN", self["gain"], "Gain")

        if self.has_key("datetime"):
            header.update("DATETIME", self["datetime"].isoformat(), "Date and time the frame was taken")

        if self.has_key("ro_mode"):
            header.update("ROMODE", self["ro_mode"], "Read-out mode")

        if self.has_key("camera"):
            header.update("CAMERA", self["camera"], "Camera ID")

        # Use non-special key COMMNT to work around bug in pyfits
        if self.has_key("comment"):
            header.update("COMMNT", self["comment"], "Frame comment")

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v



class FrameSetInfo(FrameInfo):
    framenumsep = "##"
    _allowed_keys = ("temperature", "exposure", "gain", "ro_mode", "set_comment", "camera", "datetime", "comments", "paths")
    _allowed_types = (float,        float,      str,    str,        str,          str,       list,      list,       list)

    def __getitem__(self, key):
        if key == "path":
            pref = os.path.commonprefix(self["paths"])
            tok = pref.split(self.framenumsep)
            if len(tok) == 2:
                return tok[0]
            else:
                raise KeyError("No valid path could be created")
        return dict.__getitem__(self, key)


    def update_fits_header(self, header, image=-1):
        """Update given FITS header.

        If `image` is given, assume to update the header data for image
        `image`. Otherwise (default) the primary header data is updated.
        """
        if image < 0:
            if self.has_key("temperature"):
                header.update("TEMP", self["temperature"], "Detector temperature")

            if self.has_key("exposure"):
                header.update("EXPOSURE", self["exposure"], "Exposure time in secs")

            if self.has_key("gain"):
                header.update("GAIN", self["gain"], "Gain")

            if self.has_key("datetime"):
                header.update("DATETIME", self["datetime"].isoformat(), "Date and time the frame was taken")

            if self.has_key("set_comment"):
                header.update("SETCOMM", self["set_comment"], "Data set comment")

            if self.has_key("ro_mode"):
                header.update("ROMODE", self["ro_mode"], "Read-out mode")
        else:
            try:
                cam = self.get("camera", [])[image]
            except IndexError:
                cam = ""
            try:
                com = self.get("comments", [])[image]
            except IndexError:
                com = ""

            if cam:
                header.update("CAMERA", cam, "Camera ID")
            if com:
                header.update("COMMNT", com, "Frame comment")



def QE_measurement_XEVA(abspath):
    """
    Format
    .../YYYY-MM-DD .../Filter ExpostimeUnit TempUnit/frame#.png
    """
    setdir = os.path.dirname(abspath)
    daydir = os.path.dirname(setdir)

    fi = FrameInfo()
    fi["camera"] = "XEVA-1041"

    m = tokenize_basename(setdir)
    comment = m["comment"].strip()
    if comment.startswith("XEVA"):
        comment = comment[4:].strip()
    fi["comment"] = comment

    fi["exposure"] = _convert_time2seconds(m["exposure"])
    fi["temperature"] = _convert_temperature2kelvin(m["temperature"])

    # parse date
    try:
        fi["datetime"] = get_date(os.path.basename(daydir))
    except ValueError as e:
        raise ValueError("Failed to tokenize date for %s: %s" % (abspath, e))

    return fi



class TarFile(object):
    def __init__(self, tarpath, filepath=None):
        self.tarpath = tarpath
        self.filepath = filepath

    @property
    def setinfo(self):
        fsi = FrameSetInfo()
        fsi["camera"] = "XEVA-1041"

        # Extract set comment, exposure and temperature from tar file-name.
        try:
            m = tokenize_basename(self.tarpath)
        except InfoError:
            pass
        else:
            comment = m["comment"].strip()
            if comment.startswith("XEVA"):
                comment = comment[4:].strip()
            fsi["set_comment"] = comment

            fsi["exposure"] = _convert_time2seconds(m["exposure"])
            fsi["temperature"] = _convert_temperature2kelvin(m["temperature"])

        # Initialize per-frame data containers.
        fsi["datetime"] = []
        fsi["comments"] = []
        fsi["paths"] = []

        return fsi

    @property
    def frameinfo(self):
        if self.filepath is None:
            raise ValueError("filepath must be set")

        fi = FrameInfo()
        fi["camera"] = "XEVA-1041"

        d, n = os.path.split(self.filepath)

        try:
            framenum = int(os.path.splitext(n)[0].split("_")[-1])
        except StandardError as err:
            raise ValueError("Failed to extract frame number: %s" % err)

        fi["comment"] = "Frame #%d" % framenum
        try:
            m = tokenize_basename(d)
            fi["exposure"] = _convert_time2seconds(m["exposure"])
            fi["temperature"] = _convert_temperature2kelvin(m["temperature"])
        except InfoError:
            pass

        fi["path"] = "%s#%s" % (self.tarpath, self.filepath)

        # extract the date
        tardir = os.path.dirname(self.tarpath)
        part = os.path.basename(tardir)
        try:
            fi["datetime"] = get_date(part)
        except ValueError:
            pass
        # except ValueError as e:
            # raise InfoError("Failed to tokenize date for %s: %s" % (self.tarpath, e))

        return fi



def mk_key_function(*keys):
    """Make a key-function to sort/group lists of frames (or other objects).

    Supported keys are::
        - all allowed keys of `FrameInfo`
        - `"dir"`: The sub-directory of the frame file.
        - Callables.

    Examples
    --------
    >>> # Sort frames by exposure
    >>> frames = sorted(frames, key=mk_key_function("exposure"))

    >>> # Sort frames by containing directory (i.e. the data set)
    >>> frames = sorted(frames, key=mk_key_function("dir"))

    >>> # Group frames by exposure
    >>> key_fn = mk_key_function("exposure")
    >>> frames = sorted(frames, key=key_fn)
    >>> grouped = itertools.groupby(frames, key_fn)

    >>> # Sort frames by their mean value and exposure
    >>> frames = sorted(frames,
    ...                 key=mk_key_function(lambda f: f.mean(),
    ...                                     "exposure"))
    """
    if len(keys) == 0:
        raise ValueError("At least one key must be given!")

    fcns = []
    for key in keys:
        if key in FrameInfo._allowed_keys:
            def fcn(frame):
                try:
                    return frame.info[key]
                except (AttributeError, KeyError):
                    return None
            fcns.append(fcn)

        elif key == "dir":
            def dir_fcn(frame):
                try:
                    return frame.info["path"].split(os.path.sep)[-2]
                except (AttributeError, KeyError):
                    return None
            fcns.append(dir_fcn)

        elif callable(key):
            fcns.append(key)

        elif isinstance(key, str):
            def fcn(frame):
                try:
                    return frame.info[key]
                except (AttributeError, KeyError):
                    return None
            fcns.append(fcn)

        else:
            raise ValueError("Could not construct key-function for key %s" % (key,))

    fcns = tuple(fcns)
    def key_fcn(frame):
        return tuple(f(frame) for f in fcns)
    return key_fcn
