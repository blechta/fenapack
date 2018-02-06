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


class PCDAssembler(object):
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

        All the forms are wrapped using :py:class:`PCDForm` so that each of
        them can be endowed with additional set of properties.
        By default, ``mp``, ``mu``, and ``ap`` are assumed to be constant
        if the preconditioner is used repeatedly in some outer iterative
        process (e.g Newton-Raphson method, time-stepping).
        As such, the corresponding operators are assembled only once.
        On the other hand, ``fp`` and ``kp`` are updated in every
        outer iteration. This default setting can be changed by accessing
        a :py:class:`PCDForm` instance via :py:meth:`PCDAssembler.get_pcd_form`.
        """

        # Assembler for the linear system of algebraic equations
        self.assembler = SystemAssembler(a, L, bcs)

        # Assembler for preconditioner
        if a_pc is not None:
            self.assembler_pc = SystemAssembler(a_pc, L, bcs)
        else:
            self.assembler_pc = None

        # Store bcs
        self._bcs_pcd = bcs_pcd

        # Store and initialize forms
        self._forms = {
            "L": PCDForm(L),
            "ap": PCDForm(ap, const=True),
            "mp": PCDForm(mp, const=True),
            "mu": PCDForm(mu, const=True),
            "fp": PCDForm(fp),
            "kp": PCDForm(kp),
        }


    def get_pcd_form(self, key):
        """Return form wrapped in :py:class:`PCDForm`."""
        form = self._forms.get(key)
        if form is None:
            raise AttributeError("Form '%s' requested by PCD not available" % key)
        assert isinstance(form, PCDForm)
        return form


    def get_dolfin_form(self, key):
        """Return form as :py:class:`dolfin.Form` or :py:class:`ufl.Form`."""
        return self.get_pcd_form(key).dolfin_form()


    def function_space(self):
        return self.get_dolfin_form("L").arguments()[0].function_space()


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
        assembler = SystemAssembler(self.get_dolfin_form("ap"),
                                    self.get_dolfin_form("L"),
                                    self.pcd_bcs())
        assembler.assemble(Ap)


    def mp(self, Mp):
        assemble(self.get_dolfin_form("mp"), tensor=Mp)


    def mu(self, Mu):
        assemble(self.get_dolfin_form("mu"), tensor=Mu)


    def fp(self, Fp):
        assemble(self.get_dolfin_form("fp"), tensor=Fp)


    def kp(self, Kp):
        assemble(self.get_dolfin_form("kp"), tensor=Kp)


    # FIXME: Naming
    def pcd_bcs(self):
        try:
            assert self._bcs_pcd is not None
        except (AttributeError, AssertionError):
            raise AttributeError("BCs requested by PCD not available")
        return self._bcs_pcd


class PCDForm(object):
    """Wrapper for PCD operators represented by :py:class:`dolfin.Form` or
    :py:class:`ufl.Form`. This class allows to record specific properties of
    the form that can be utilized later while setting up the preconditioner.

    For example, we can specify which matrices remain constant during the outer
    iterative algorithm (e.g. Newton-Raphson method, time-stepping) and which
    matrices need to be updated in every outer iteration.
    """
    def __init__(self, form, const=False):
        """The class is initialized by a single form with default properties.

        *Arguments*
            form (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                A form to be wrapped.
            const (`bool`)
                Whether the form remains constant in outer iterations.
        """
        # Store form
        self._form = form

        # Initialize public properties
        assert isinstance(const, bool)
        self.constant = const

    def dolfin_form(self):
        return self._form

    def is_constant(self):
        return self.constant
