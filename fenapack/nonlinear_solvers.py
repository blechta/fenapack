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

from fenapack._field_split_utils import SubfieldBC

__all__ = ['NewtonSolver', 'PCDProblem']


# FIXME: Rename, this is specialized
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
        # C++ references passed in do not have Python context
        linear_solver = self._solver
        nonlinear_problem = self._problem

        P = A if P.empty() else P

        linear_solver.set_operators(A, P, pcd_problem=nonlinear_problem)



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

        # Store forms/bcs for later
        self.forms = {
            "F": F,
            "ap": ap,
            "mp": mp,
            "mu": mu,
            "fp": fp,
            "kp": kp,
        }
        self._bcs_pcd = bcs_pcd

        # Store mpi comm
        self._mpi_comm = F.ufl_domain().ufl_cargo().mpi_comm()


    def get_form(self, key):
        try:
            return self.forms[key]
        except KeyError:
            raise AttributeError("From '%s' requested by not available" % key)


    def F(self, b, x):
        self.assembler.assemble(b, x)


    def J(self, A, x):
        self.assembler.assemble(A)


    def J_pc(self, P, x):
        if self.assembler_pc is not None:
            self.assembler_pc.assemble(P)


    def ap(self, Ap):
        # FIXME: Do we want to have this logic here?
        if Ap.empty():
            ap = self.get_form("ap")
            F = self.get_form("F")  # Dummy form
            bcs_pcd = self.pcd_bcs()
            assembler = dolfin.SystemAssembler(ap, F, bcs_pcd)
            assembler.assemble(Ap)


    def mp(self, Mp):
        dolfin.assemble(self.get_form("mp"), tensor=Mp)


    def mu(self, Mu):
        dolfin.assemble(self.get_form("mu"), tensor=Mu)


    def fp(self, Fp):
        dolfin.assemble(self.get_form("fp"), tensor=Fp)


    def kp(self, Kp):
        dolfin.assemble(self.get_form("kp"), tensor=Kp)


    # FIXME: Naming
    def pcd_bcs(self):
        try:
            assert self._bcs_pcd is not None
            return self._bcs_pcd
        except (AttributeError, AssertionError):
            raise AttributeError("PCD BCs requested by not available")


    def mpi_comm(self):
        return self._mpi_comm



# backend
class _PCDProblem(PCDProblem):
    def __init__(self):
        pass


    @classmethod
    def from_pcd_problem(cls, pcd_problem):
        self = cls()
        self.__dict__ = pcd_problem.__dict__
        return self


    def setup_ksp_Ap(self, ksp):
        self._setup_ksp_once(ksp, self.ap)


    def setup_ksp_Mp(self, ksp):
        self._setup_ksp_once(ksp, self.mp)


    def setup_mat_Kp(self, mat):
        # FIXME: Parametrize me
        return self._setup_Kp_shallow(mat)
        #return self._setup_Kp_deep(mat)


    def apply_pcd_bcs(self, vec):
        # FIXME: interface for general bc tweaks of matrices?
        self._apply_bcs(vec, self.pcd_bcs)


    # Factor two following function from Kp

    def _setup_Kp_shallow(self, mat):
        # FIXME: Improve this confusing logic and naming?!
        scratch = getattr(self, "_Kp_scratch", None)
        mat, scratch = self._assemble_mat_shallow(mat, self.kp, scratch)
        self._Kp_scratch = scratch
        return mat

    def _setup_Kp_deep(self, mat):
        return self._assemble_mat_deep(mat, self.kp)


    def _apply_bcs(self, vec, bcs_getter):
        # FIXME: Improve this confusing logic and naming?!
        subbcs = getattr(self, "_subbcs", None)
        if subbcs is None:
            bcs = bcs_getter()
            bcs = [bcs] if isinstance(bcs, dolfin.DirichletBC) else bcs
            subbcs = [SubfieldBC(bc, self._is1) for bc in bcs]
        self._subbcs = subbcs

        for bc in subbcs:
            bc.apply(vec)


    def _setup_ksp_once(self, ksp, assembler_func):
        mat = ksp.getOperators()[0]
        # FIXME: This logic that it is created once should be visible
        #        in higher level, not in these internals
        if mat.type is None:
            # FIXME: Could have shared work matrix
            A = dolfin.PETScMatrix(mat.comm)
            assembler_func(A)
            mat = self._get_deep_submat(A, None, self._is1)
            mat.setOption(PETSc.Mat.Option.SPD, True)  # FIXME: Can't do this for Kp/Fp
            ksp.setOperators(mat, mat)
            assert ksp.getOperators()[0].type is not None


    def _assemble_mat_shallow(self, petsc_mat, assembler_func, dolfin_mat=None):
        # Assemble dolfin_mat everytime
        dolfin_mat = dolfin_mat or dolfin.PETScMatrix(self.mpi_comm())
        assembler_func(dolfin_mat)

        # FIXME: This logic that it is created once should be visible
        #        in higher level, not in these internals
        # Create shallow submatrix once
        if petsc_mat is None or petsc_mat.type is None:
            petsc_mat = self._get_shallow_submat(dolfin_mat, petsc_mat, self._is1)
            assert petsc_mat.type is not None

        # Return allocated mats so that client can strore it
        return petsc_mat, dolfin_mat


    def _assemble_mat_deep(self, petsc_mat, assembler_func):
        # FIXME: Could have shared work matrix
        dolfin_mat = dolfin.PETScMatrix(self.mpi_comm())
        assembler_func(dolfin_mat)
        return self._get_deep_submat(dolfin_mat, petsc_mat, self._is1)


    @staticmethod
    def _get_deep_submat(dolfin_mat, petsc_submat, iset):
        #if petsc_submat is not None and petsc_submat.type is None:
        #    petsc_submat.setSizes(((iset.size, iset.size), (iset.size, iset.size)))
        return dolfin_mat.mat().getSubMatrix(iset, iset, submat=petsc_submat)


    @staticmethod
    def _get_shallow_submat(dolfin_mat, petsc_submat, iset):
        if petsc_submat is None:
            petsc_submat = PETSc.Mat().create(iset.comm)
        return petsc_submat.createSubMatrix(dolfin_mat.mat(), iset, iset)


    def set_is1(self, is1):
        self._is1 = is1
