#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  setuptools import setup
import re, os, glob

version = re.findall('__version__ = "(.*)"',
                     open('fenapack/__init__.py', 'r').read())[0]

CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python :: 3
Programming Language :: C++
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

demofiles = (
      glob.glob(os.path.join("demo", "**", "*.py"), recursive=True)
    + glob.glob(os.path.join("demo", "**", "*.rst"), recursive=True)
    + glob.glob(os.path.join("demo", "data", "*.xml"))
)
data_files=[(os.path.join("share", "fenapack", os.path.dirname(f)), [f])
            for f in demofiles]

with open("README.rst", "r") as f:
    long_description = f.read()

setup(name="fenapack",
      version=version,
      author="Jan Blechta",
      author_email="blechta@karlin.mff.cuni.cz",
      url="https://github.com/blechta/fenapack",
      description="FENaPack: FEniCS Navier-Stokes preconditioning package",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      classifiers=classifiers,
      packages=["fenapack"],
      package_dir={"fenapack": "fenapack"},
      package_data={"fenapack": ["*.h"]},
      data_files=data_files)
