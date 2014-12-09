# SPE reader module. Based on a Python module by Vincent Favre-Nicolin (See
# <http://mail.scipy.org/pipermail/scipy-user/2007-November/014673.html>)
# and the WinView documentation.
#
# Copyright 2010-2012  Jan Eike von Seggern jan.eike.von.seggern@desy.de

import datetime
import numpy as np


_month_names = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ["Jan", "Feb", "Mae", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"],
        ]


class SPEReader(object):
    """Reader for WinView/32 .spe files.

    Attributes read::

        datetime : `datetime.datetime`
            The local time at which the frame was taken.

        exposure : `float`
            The exposure time in seconds.

        dtype : numpy data type
            The data type stored in the frame, e.g. `numpy.int16` or
            `numpy.float`.

        readoutrate : `str`
            The read-out rate used ("2MHz" or "100kHz").
    """

    "Dictonary of byte offsets of header data."
    _offsets = {
               "DATE" : 20,
               "TIME" : 172,
               "EXPOSURE" : 10,
               "XDIM" : 42,
               "YDIM" : 656,
               "TEMP" : 36,
               "DATATYPE" : 108,
               "NUMBEROFFRAMES" : 1446,
               "DATA" : 4100,
               "ADCOFFSET" : 188,
               "READOUTRATE" : 190,
               "SHUTTERCONTROL" : 50,
               "SHUTTERCOMP" : 1476
               }
    _datatypes = {0 : np.float,
                  1 : np.int32,
                  2 : np.int16,
                  3 : np.uint16 }
    _readoutrates = {"\x0c\x00" : "2MHz",
                     "\x06\x00" : "100kHz"}
    # TODO: find the correct values.
    _shuttercontrols = {}

    def __init__(self, path=None, file=None):
        if path is None and file is None:
            raise ValueError("path or file expected")
        elif path is not None and file is not None:
            raise ValueError("only path or file may be specified")
        elif path is not None:
            self._file = open(path, "rb")
        else:
            self._file = file

        # the np dtype
        self._dtype = None
        self._datetime = None
        self._num_frames = None
        # (xdim, ydim)
        self._shape = None
        self._data = None
        self._temp = None
        self._exposure = None
        self._readoutrate = None
        self._adcoffset = None
        self._shuttercontrol = None
        self._shuttercompensation = None

        # protect seek'ed positions (see self._seek, self._done)
        self._locked = None

    def __str__(self):
        return "<%s for file %s>" % (self.__class__.__name__, self._file.name)

    def _seek(self, where):
        if self._locked is not None:
            raise RuntimeError("Locked at %s" % self._locked)

        self._file.seek(self._offsets[where])
        self._locked = where

    def _done(self, where):
        if self._locked != where:
            raise RuntimeError("Locked at %s. Cannot release lock for %s." % (self._locked, where))
        self._locked = None

    @property
    def exposure(self):
        if self._exposure is None:
            self._seek("EXPOSURE")
            self._exposure = np.fromfile(self._file, np.float32, 1)[0]
            self._done("EXPOSURE")
        return self._exposure

    @property
    def num_frames(self):
        if self._num_frames is None:
            self._seek("NUMBEROFFRAMES")
            self._num_frames = np.fromfile(self._file, np.int16, 1)[0]
            self._done("NUMBEROFFRAMES")
        return self._num_frames

    @property
    def shape(self):
        """The (y, x) shape of one frame."""
        if self._shape is None:
            self._seek("XDIM")
            xdim = int(np.fromfile(self._file, np.int16, 1)[0])
            self._done("XDIM")

            self._seek("YDIM")
            ydim = int(np.fromfile(self._file, np.int16, 1)[0])
            self._done("YDIM")

            self._shape = (ydim, xdim)
        return self._shape

    @property
    def temp(self):
        """The detector temperature in degree Celsius."""
        if self._temp is None:
            self._seek("TEMP")
            self._temp = np.fromfile(self._file, np.float32, 1)[0]
            self._done("TEMP")
        return self._temp

    @property
    def data(self):
        if self._data is None:
            self._load_data()
        return self._data

    def _load_data(self):
        dtype = self.dtype
        frame_shape = self.shape
        num_frames = self.num_frames
        data_points = frame_shape[0] * frame_shape[1] * num_frames
        data_shape = (num_frames, frame_shape[0], frame_shape[1])

        self._seek("DATA")
        self._data = np.fromfile(self._file, dtype, data_points)
        self._done("DATA")
        self._data.shape = data_shape

    @property
    def dtype(self):
        if self._dtype is None:
            self._seek("DATATYPE")
            self._dtype = self._datatypes[np.fromfile(self._file, np.int16, 1)[0]]
            self._done("DATATYPE")

        return self._dtype

    @property
    def datetime(self):
        if self._datetime is None:
            self._seek("DATE")
            date = self._file.read(9)
            self._done("DATE")
            self._seek("TIME")
            time = self._file.read(6)
            self._done("TIME")

            day = int(date[:2])
            month = -1
            for months in _month_names:
                try:
                    month = months.index(date[2:5]) + 1
                    break
                except ValueError:
                    pass
            if month <= 0:
                print("Error: Could not parse month name in SPE header: %s" % (date[2:5]))
                month = 1
            year = int(date[5:])

            hours = int(time[:2])
            mins = int(time[2:4])
            secs = int(time[4:6])
            # msecs = float(time[7:])

            self._datetime = datetime.datetime(year, month, day, hours, mins, secs)

        return self._datetime

    @property
    def readoutrate(self):
        if self._readoutrate is None:
            self._seek("READOUTRATE")
            self._readoutrate = self._readoutrates[self._file.read(2)]
            self._done("READOUTRATE")
        return self._readoutrate

    @property
    def adcoffset(self):
        if self._adcoffset is None:
            self._seek("ADCOFFSET")
            self._adcoffset = np.fromfile(self._file, np.int16, 1)[0]
            self._done("ADCOFFSET")
        return self._adcoffset

    @property
    def shutter_control(self):
        return NotImplemented
        if self._shuttercontrol is None:
            self._seek("SHUTTERCONTROL")
            self._shuttercontrol = self._shuttercontrols[self._file.read(2)]
            self._done("SHUTTERCONTROL")
        return self._shuttercontrol

    @property
    def shutter_compensation(self):
        if self._shuttercompensation is None:
            self._seek("SHUTTERCOMP")
            self._shuttercompensation = np.fromfile(self._file, np.float16, 1)[0]
            self._done("SHUTTERCOMP")
        return self._shuttercompensation

    def as_primary_hdu(self):
        import pyfits
        hdu = pyfits.PrimaryHDU(self.data)
        hdu.header.update("DATETIME", self.datetime.isoformat(),
                          "Date and time frame was taken")
        hdu.header.update("TEMP", self.temp, "Detector temperature")
        hdu.header.update("EXPOSURE", self.exposure, "Exposure time")
        hdu.header.update("READRATE", self.readoutrate, "Read-out rate")
        return hdu

    def as_TH2I(self, name="image", title="converted SPE frame"):
        import ROOT
        nx, ny = self.shape
        histo = ROOT.TH2I(name, title, nx, 0, nx, ny, 0, ny)
        for ix in range(nx):
            for iy in range(ny):
                histo.SetBinContent(ix+1, iy+1, self.data[ix, iy])
        return histo


if __name__  ==  "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("Usage:  %s  FILE.spe [...]" % sys.argv[0])
    for path in sys.argv[1:]:
        spe = SPEReader(path)
        print("Frame taken at:", spe.datetime)
        print("Read-out rate:", spe.readoutrate)
        print("Datatype:", spe.dtype)
        print("Temp:", spe.temp)
        print("Exposure time:", spe.exposure)
        print("Data shape:", spe.data.shape)
        print("Max value:", spe.data.max())
