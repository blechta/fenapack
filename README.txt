=======================================================
FENaPack - FEniCS Navier-Stokes preconditioning package
=======================================================

FENaPack is a package implementing preconditioners for Navier-Stokes
problem using FEniCS and PETSc packages. In particular, PCD
(pressure-convection-diffussion) preconditioner [1]_ is implemented.

.. [1] Elman H. C., Silvester D. J., Wathen A. J., *Finite Elements and Fast
       Iterative Solvers: With Application in Incompressible Fluid Dynamics*.
       Oxford University Press 2005. 2nd edition 2014.


Usage
=====

To use FENaPack you need FEniCS (version 1.7.0 or higher) compiled with PETSc
and petsc4py. To be able to import FENaPack functions, update your PYTHONPATH
using 'fenapack.conf' and do the usual setup needed to run FEniCS.

Meshes for running demos can be downloaded from FEniCS project
website by executing 'download-meshes' script. Demos can be run
by navigating to a particular demo directory and typing::

  NP=4
  mpirun -n $NP python demo_foo-bar.py [-h]


Authors
=======

- Jan Blechta <blechta@karlin.mff.cuni.cz>
- Martin Rehor <rehor@karlin.mff.cuni.cz>


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
