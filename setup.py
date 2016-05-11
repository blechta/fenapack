#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
import re, os, glob

version = re.findall('__version__ = "(.*)"',
                     open('fenapack/__init__.py', 'r').read())[0]

CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python
Programming Language :: C++
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

demofiles = glob.glob(os.path.join("demo", "*", "*.py"))

setup(name="FENaPack",
      version=version,
      author="Jan Blechta, Martin Řehoř",
      author_email="blechta@karlin.mff.cuni.cz",
      url="http://github.com/blechta/fenapack",
      description="FEniCS Navier-Stokes preconditioning package",
      classifiers=classifiers,
      license="GNU LGPL v3 or later",
      packages=["fenapack"],
      package_dir={"fenapack": "fenapack"},
      package_data={"fenapack": ["*.h"]},
      data_files=[(os.path.join("share", "fenapack", os.path.dirname(f)), [f])
                  for f in demofiles],
    )
