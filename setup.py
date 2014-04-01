#! /usr/bin/env python
from distutils.core import setup

setup(name='ALPS-CCD',
      version='0.0.1',
      description='Tools to read and process CCD images',
      author='Jan Eike von Seggern',
      author_email='jan.eike.von.seggern@desy.de',
      packages=['ccd', 'ccd.analysis', 'ccd.analysis.hotpixels', 'ccd.io'],
      requires=['numpy', 'scipy', 'matplotlib']
      )
