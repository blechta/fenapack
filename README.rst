*******************************************************
FENaPack - FEniCS Navier-Stokes preconditioning package
*******************************************************

.. image:: https://travis-ci.org/blechta/fenapack.svg?branch=master
    :target: https://travis-ci.org/blechta/fenapack


FENaPack is a package implementing preconditioners for Navier-Stokes
problem using FEniCS and PETSc packages. In particular, PCD
(pressure-convection-diffussion) preconditioner [1]_ is implemented.

.. [1] Elman H. C., Silvester D. J., Wathen A. J., *Finite Elements and Fast
       Iterative Solvers: With Application in Incompressible Fluid Dynamics*.
       Oxford University Press 2005. 2nd edition 2014.


Usage
=====

To use FENaPack matching version of FEniCS (version |version|) compiled with
PETSc and petsc4py is needed. To use FENaPack either install it by standard
procedures or ``source fenapack.conf``.

Meshes for running demos can be downloaded from the FEniCS project
website by executing ``download-meshes`` script. Demos can be run
by navigating to a particular demo directory and typing::

  NP=4
  mpirun -n $NP python demo_foo-bar.py [-h]

Full documentation is available at https://blechta.github.io/fenapack/.


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


Links
=====

- Homepage https://github.com/blechta/fenapack
- Testing https://travis-ci.org/blechta/fenapack
- Documentation https://blechta.github.io/fenapack/
- Bug reports https://github.com/blechta/fenapack/issues
