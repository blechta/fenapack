*******************************************************
FENaPack - FEniCS Navier-Stokes preconditioning package
*******************************************************

.. image:: https://circleci.com/gh/blechta/fenapack.svg?style=svg
    :target: https://circleci.com/gh/blechta/fenapack


FENaPack is a package implementing preconditioners for Navier-Stokes
problem using FEniCS and PETSc packages. In particular, variants of PCD
(pressure-convection-diffussion) preconditioner from [1]_, [2]_ are
implemented.

.. [1] Elman H. C., Silvester D. J., Wathen A. J., *Finite Elements and Fast
       Iterative Solvers: With Application in Incompressible Fluid Dynamics*.
       Oxford University Press 2005. 2nd edition 2014.

.. [2] Olshanskii M. A., Vassilevski Y. V., *Pressure Schur complement
       preconditioners for the discrete Oseen problem*.
       SIAM J. Sci. Comput., 29(6), 2686-2704. 2007.


Usage
=====

To use FENaPack matching version of FEniCS (version 2018.1.0)
compiled with PETSc, petsc4py and mpi4py is needed. Note that
FENaPack uses same version numbering as FEniCS and follows
its release schedule with a short lag.

To install FENaPack from source do::

  pip3 install [--user|--prefix=...] [-e] .

in the source/repository root dir.  Editable install using ``-e``
allows to use FENaPack directly from source directory while
editing it which is suitable for development.

You can install latest FENaPack release form PyPI::

  pip3 install [--user|--prefix=...] fenapack

or install latest development version from Github::

  pip3 install [--user|--prefix=...] git+https://github.com/blechta/fenapack

To start experimenting::

  cd demo/navier-stokes-pcd
  python3 demo_navier-stokes-pcd.py --help
  python3 demo_navier-stokes-pcd.py [opts]
  mpirun -n 16 python3 demo_navier-stokes-pcd.py [opts]

Full documentation is available at https://fenapack.readthedocs.io/.


Authors
=======

- Jan Blechta <blechta@karlin.mff.cuni.cz>
- Martin Řehoř <rehor@karlin.mff.cuni.cz>


License
=======

FENaPack is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FENaPack is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FENaPack. If not, see <http://www.gnu.org/licenses/>.


Acknowledgement
===============

This work was supported by The Ministry of Education, Youth and Sports from the
Large Infrastructures for Research, Experimental Development and Innovations
project „IT4Innovations National Supercomputing Center – LM2015070“.


Links
=====

- Homepage https://github.com/blechta/fenapack
- Testing https://circleci.com/gh/blechta/fenapack
- Documentation https://fenapack.readthedocs.io/
- Bug reports https://github.com/blechta/fenapack/issues
- PyPI home https://pypi.org/project/fenapack
