alps-ccd: The ALPS CCD toolbox
==============================

This package contains tools to read and analyze CCD frames. It has been tested
on Python 2.7 and Python 3.4+.

Required Python packages are:
  - numpy
  - scipy
  - matplotlib


Detailed installation on Ubuntu
-------------------------------

This installation instructions describe how to create a new virtualenv and
install the package into this.

1. Install the following Ubuntu packages:
    - python-numpy
    - python-scipy
    - python-matplotlib
    - ipython
    - python-virtualenv
    - git

2. Prepare install directory:

   We will download and install my Python code into a "special" directory (a
   virtual environment) in order to not pollute your hard disk.

   2.1 Create "virtualenv" named ALPS (or any other name and anywhere you like) with:

       ::

          virtualenv --system-site-packages --clear ALPS

       and activate it for the current shell session::

          source ALPS/bin/activate

   2.2 Prepare download directory:

       ::

          mkdir ALPS/src
      
       and go there::

          cd ALPS/src

3. Download and install my packages from GitHub

  3.1 Download
    ::

      git clone https://github.com/eikevons/alps-ccd.git
      git clone https://github.com/eikevons/plttools.git

  3.1 Install the packages in the "ALPS" virtualenv (from inside
    `.../ALPS/src/`)::

      cd ../lib/python2.7/site-packages
      ln -s ../../../src/alps-ccd/ccd/ .
      ln -s ../../../src/plttools/plttools/ .

    (Note: The more "standard" way of installing Python packages is to use the
    `setup.py` script. But in our case, the code might change frequently and
    linking as above ensures that the most recent version is used at all times.)

4. Test that the install worked.

    4.1 Start IPython and try to load my modules::

      ipython
      In [1]: import ccd, ccd.io, ccd.analysis, ccd.analysis.hotpixels
      In [2]: import plttools
