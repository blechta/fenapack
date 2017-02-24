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

import dolfin
from petsc4py import PETSc

__all__ = ['NewtonSolver', 'PCDProblem']


class NewtonSolver(dolfin.NewtonSolver):

    def __init__(self, solver):
        comm = solver.mpi_comm()
        factory = dolfin.PETScFactory.instance()
        dolfin.NewtonSolver.__init__(self, comm, solver, factory)
        self._solver = solver

    def solve(self, problem, x):
        self._problem = problem
        r = dolfin.NewtonSolver.solve(self, problem, x)
        del self._problem
        return r

    def solver_setup(self, A, P, nonlinear_problem, iteration):
        linear_solver = self._solver
        nonlinear_problem = self._problem

        if P.empty():
            P = A

        schur_approx = {}

        # FIXME: Clean this up! This is a mess!

        # Prepare matrices guaranteed to be constant during iterations once
        if iteration == 0:
            schur_approx["bcs"] = nonlinear_problem._bcs_pcd
            for key in ["Ap", "Mp", "Mu"]:
                mat_object = getattr(nonlinear_problem, "_mat"+key)
                try:
                    getattr(nonlinear_problem, key.lower())(mat_object) # assemble
                except AttributeError:
                    pass
                else:
                    schur_approx[key] = mat_object

        # Assemble non-constant matrices everytime
        for key in ["Fp", "Kp"]:
            mat_object = getattr(nonlinear_problem, "_mat"+key)
            try:
                getattr(nonlinear_problem, key.lower())(mat_object) # assemble
            except AttributeError:
                pass
            else:
                schur_approx[key] = mat_object

        # Pass assembled operators and bc to linear solver
        linear_solver.set_operators(A, P, **schur_approx)



class PCDProblem(dolfin.NonlinearProblem):
    """Class for interfacing with :py:class:`NewtonSolver`."""
    def __init__(self, F, bcs, J, J_pc=None,
                 mp=None, mu=None, ap=None, fp=None, kp=None, bcs_pcd=[]):
        """Return subclass of :py:class:`dolfin.NonlinearProblem` suitable
        for :py:class:`NewtonSolver` based on
        :py:class:`fenapack.field_split.FieldSplitSolver` and PCD
        preconditioners from :py:class:`fenapack.field_split`.

        *Arguments*
            F (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Linear form representing the equation.
            bcs (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Boundary conditions applied to ``F``, ``J``, and ``J_pc``.
            J (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing system Jacobian.
            J_pc (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing Jacobian optionally passed to
                preconditioner instead of ``J``. In case of PCD, stabilized
                00-block can be passed to 00-KSP solver.
            mp, mu, ap, fp, kp (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear forms which (some of them) might be used by a
                particular PCD preconditioner. Typically they represent "mass
                matrix" on pressure, "mass matrix" on velocity, minus Laplacian
                operator on pressure, pressure convection-diffusion operator,
                and pressure convection operator respectively.

                ``mp``, ``mu``, and ``ap`` are assumed to be constant during
                subsequent non-linear iterations and are assembled only once.
                On the other hand, ``fp`` and ``kp`` are updated in every
                iteration.
            bcs_pcd (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Artificial boundary conditions used by PCD preconditioner.

        All the arguments should be given on the common mixed function space.
        """

        dolfin.NonlinearProblem.__init__(self)

        # Assembler for Newton/Picard system
        # FIXME: Does it get broken for Oseen system?
        self.assembler = dolfin.SystemAssembler(J, F, bcs)

        # Assembler for preconditioner
        if J_pc is not None:
            self.assembler_pc = dolfin.SystemAssembler(J_pc, F, bcs)
        else:
            self.assembler_pc = None

        self._F = F
        self._mp = mp
        self._mu = mu
        self._ap = ap
        self._fp = fp
        self._kp = kp
        self._bcs_pcd = bcs_pcd

        # Matrices used to assemble parts of the preconditioner
        # NOTE: Some of them may be unused.
        comm = F.ufl_domain().ufl_cargo().mpi_comm()
        self._matMp = dolfin.PETScMatrix(comm)
        self._matMu = dolfin.PETScMatrix(comm)
        self._matAp = dolfin.PETScMatrix(comm)
        self._matFp = dolfin.PETScMatrix(comm)
        self._matKp = dolfin.PETScMatrix(comm)


    def F(self, b, x):
        self.assembler.assemble(b, x)


    def J(self, A, x):
        self.assembler.assemble(A)


    def J_pc(self, P, x):
        if self.assembler is not None:
            self.assembler.assemble(P)


    def mp(self, Mp):
        # Mp ... pressure mass matrix
        self._check_attr("mp")
        dolfin.assemble(self._mp, tensor=Mp)


    def mu(self, Mu):
        # Mu ... velocity mass matrix
        self._check_attr("mu")
        dolfin.assemble(self._mu, tensor=Mu)


    def ap(self, Ap):
        # Ap ... pressure Laplacian matrix
        self._check_attr("ap")

        if Ap.empty():
            assembler = dolfin.SystemAssembler(self._ap, self._F, self._bcs_pcd)
            assembler.assemble(Ap)
            Ap.mat().setOption(PETSc.Mat.Option.SPD, True)


    def fp(self, Fp):
        # Fp ... pressure convection-diffusion matrix
        self._check_attr("fp")
        dolfin.assemble(self._fp, tensor=Fp)


    def kp(self, Kp):
        # Kp ... pressure convection matrix
        self._check_attr("kp")
        dolfin.assemble(self._kp, tensor=Kp)


    def _check_attr(self, attr):
        if getattr(self, "_"+attr) is None:
            raise AttributeError("Keyword argument '%s' of 'PCDProblem' object"
                                 " has not been set." % attr)
