# Copyright (C) 2015-2017 Jan Blechta and Martin Rehor
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

"""This module provides subclasses of DOLFIN interface for
solving non-linear problems suitable for use with fieldsplit
preconditioned Krylov methods
"""

from dolfin import NewtonSolver, PETScFactory, NonlinearProblem

from fenapack.assembling import PCDAssembler


class PCDNewtonSolver(NewtonSolver):
    """Newton solver suitable for use with
    :py:class:`fenapack.field_split.PCDKrylovSolver`.
    """

    def __init__(self, solver, pcd_pc_class=None):
        """Initialize for a given PCD Krylov solver and preconditioner class.

        *Arguments*
            solver (:py:class:`fenapack.field_split.PCDKrylovSolver`)
               A PCD Krylov solver.
            pcd_pc_class (:py:class:`fenapack.preconditioners.BasePCDPC`)
               Representative of a class implementing PCD preconditioner.
        """

        # Initialize DOLFIN Newton solver
        comm = solver.ksp().comm.tompi4py()
        factory = PETScFactory.instance()
        super(PCDNewtonSolver, self).__init__(comm, solver, factory)

        # Store Python reference for solver setup
        self._solver = solver
        self._pcd_pc_class = pcd_pc_class


    def solve(self, problem, x):
        # Store Python reference for solver setup
        self._problem = problem

        # Solve the problem, drop the reference, and return
        r = super(PCDNewtonSolver, self).solve(problem, x)
        del self._problem
        return r


    def solver_setup(self, A, P, nonlinear_problem, iteration):
        # Only do the setup once
        # FIXME: Is this good?
        if iteration > 0 or getattr(self, "_initialized", False):
            return
        self._initialized = True

        # C++ references passed in do not have Python context
        linear_solver = self._solver
        nonlinear_problem = self._problem

        # Set operators and initialize PCD
        P = A if P.empty() else P
        linear_solver.set_operators(A, P)
        linear_solver.init_pcd(
            nonlinear_problem.pcd_assembler, self._pcd_pc_class)


    def linear_solver(self):
        return self._solver


class PCDNonlinearProblem(NonlinearProblem):
    """Class for interfacing with :py:class:`PCDNewtonSolver`."""

    def __init__(self, pcd_assembler):
        """Return subclass of :py:class:`dolfin.NonlinearProblem`
        suitable for :py:class:`NewtonSolver` based on
        :py:class:`fenapack.field_split.PCDKrylovSolver` and
        PCD preconditioners from :py:class:`fenapack.preconditioners`.

        *Arguments*
            pcd_assembler (:py:class:`fenapack.assembling.PCDAssembler`)
               A class which takes care of assembling PCD operators on demand.
        """

        assert isinstance(pcd_assembler, PCDAssembler)
        super(PCDNonlinearProblem, self).__init__()
        self.pcd_assembler = pcd_assembler

    def F(self, b, x):
        self.pcd_assembler.rhs_vector(b, x)


    def J(self, A, x):
        self.pcd_assembler.system_matrix(A)


    def J_pc(self, P, x):
        self.pcd_assembler.pc_matrix(P)
