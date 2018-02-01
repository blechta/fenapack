# Copyright (C) 2015-2018 Jan Blechta and Martin Rehor
#
# This file is part of FENaPack.
#
# FENaPack is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FENaPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FENaPack.  If not, see <http://www.gnu.org/licenses/>.

"""This module provides wrappers for assembling systems of linear algebraic
equations to be solved with the use of PCD preconditioning strategy.
These wrappers naturally provide routines for assembling preconditioning
operators themselves.
"""

from dolfin import SystemAssembler, assemble


class PCDProblem(object):
    """Base class for creating linear problems to be solved by application
    of the PCD preconditioning strategy. Users are encouraged to use this class
    for interfacing with :py:class:`fenapack.field_split.PCDKrylovSolver`.
    On request it assembles not only the individual PCD operators but also the
    system matrix and the right hand side vector defining the linear problem.
    """

    def __init__(self, a, L, bcs, a_pc=None,
                 mp=None, mu=None, ap=None, fp=None, kp=None, bcs_pcd=[]):
        """Collect individual variational forms and boundary conditions
        defining a linear problem (system matrix + RHS vector) on the one side
        and preconditioning operators on the other side.

        *Arguments*
            a (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing a system matrix.
            L (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Linear form representing a right hand side vector.
            bcs (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Boundary conditions applied to ``a``, ``L``, and ``a_pc``.
            a_pc (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing a matrix optionally passed to
                preconditioner instead of ``a``. In case of PCD, stabilized
                00-block can be passed to 00-KSP solver.
            mp, mu, ap, fp, kp (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear forms which (some of them) might be used by a
                particular PCD preconditioner. Typically they represent "mass
                matrix" on pressure, "mass matrix" on velocity, minus Laplacian
                operator on pressure, pressure convection-diffusion operator,
                and pressure convection operator respectively.
            bcs_pcd (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Artificial boundary conditions used by PCD preconditioner.

        All the arguments should be given on the common mixed function space.
        """

        # Assembler for the linear system of algebraic equations
        self.assembler = SystemAssembler(a, L, bcs)

        # Assembler for preconditioner
        if a_pc is not None:
            self.assembler_pc = SystemAssembler(a_pc, L, bcs)
        else:
            self.assembler_pc = None

        # Store forms/bcs for later use
        self.forms = {
            "L": L,
            "ap": ap,
            "mp": mp,
            "mu": mu,
            "fp": fp,
            "kp": kp,
        }
        self._bcs_pcd = bcs_pcd


    def get_form(self, key):
        form = self.forms.get(key)
        if form is None:
            raise AttributeError("Form '%s' requested by PCD not available" % key)
        return form


    def function_space(self):
        return self.forms["L"].arguments()[0].function_space()


    def rhs_vector(self, b, x=None):
        """Assemble right hand side vector ``b``.

        The version with ``x`` is suitable for use inside
        a (quasi)-Newton solver.
        """
        if x is not None:
            self.assembler.assemble(b, x)
        else:
            self.assembler.assemble(b)

    def system_matrix(self, A):
        """Assemble system matrix ``A``."""
        self.assembler.assemble(A)


    def pc_matrix(self, P):
        """Assemble preconditioning matrix ``P`` whose relevant blocks can be
        passed to actual parts of the ``KSP`` solver.
        """
        if self.assembler_pc is not None:
            self.assembler_pc.assemble(P)


    def ap(self, Ap):
        assembler = SystemAssembler(self.get_form("ap"), self.get_form("L"),
                                    self.pcd_bcs())
        assembler.assemble(Ap)


    def mp(self, Mp):
        assemble(self.get_form("mp"), tensor=Mp)


    def mu(self, Mu):
        assemble(self.get_form("mu"), tensor=Mu)


    def fp(self, Fp):
        assemble(self.get_form("fp"), tensor=Fp)


    def kp(self, Kp):
        assemble(self.get_form("kp"), tensor=Kp)


    # FIXME: Naming
    def pcd_bcs(self):
        try:
            assert self._bcs_pcd is not None
        except (AttributeError, AssertionError):
            raise AttributeError("BCs requested by PCD not available")
        return self._bcs_pcd
