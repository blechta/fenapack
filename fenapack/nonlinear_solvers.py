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

__all__ = ['PCDNewtonSolver', 'PCDProblem']


class PCDNewtonSolver(dolfin.NewtonSolver):

    def __init__(self, solver):
        # Initialize DOLFIN Newton solver
        comm = solver.mpi_comm()
        factory = dolfin.PETScFactory.instance()
        dolfin.NewtonSolver.__init__(self, comm, solver, factory)

        # Store Python reference for solver setup
        self._solver = solver


    def solve(self, problem, x):
        # Store Python reference for solver setup
        self._problem = problem

        # Solve the problem, drop the reference, and return
        r = dolfin.NewtonSolver.solve(self, problem, x)
        del self._problem
        return r


    def solver_setup(self, A, P, nonlinear_problem, iteration):
        # Only do the setup once
        if iteration > 0:
            return

        # C++ references passed in do not have Python context
        linear_solver = self._solver
        nonlinear_problem = self._problem

        # Set operators and initialize PCD
        P = A if P.empty() else P
        linear_solver.set_operators(A, P)
        #linear_solver.set_from_options()  # FIXME: Who calls this for us?
        linear_solver.init_pcd(nonlinear_problem)



# FIXME: Separate Newton and PCD part of this class. Linear PCD problem
# has nothing to do with Newton part
class PCDProblem(dolfin.NonlinearProblem):
    """Class for interfacing with not only :py:class:`NewtonSolver`."""
    # TODO: Add abstract base class with docstrings
    # TODO: What about interface
    #          pcd_problem = PCDProblem()
    #          pcd_problem.forms.F = F
    #          pcd_problem.forms.J = J
    #          ....
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
            raise AttributeError("Form '%s' requested by PCD not available" % key)


    def F(self, b, x):
        self.assembler.assemble(b, x)


    def J(self, A, x):
        self.assembler.assemble(A)


    def J_pc(self, P, x):
        if self.assembler_pc is not None:
            self.assembler_pc.assemble(P)


    def ap(self, Ap):
        assembler = dolfin.SystemAssembler(self.get_form("ap"),
                                           self.get_form("F"),
                                           self.pcd_bcs())
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
# FIXME: Here will come GenericPCDProblem
class _PCDProblem(PCDProblem):
    """Wrapper of PCDProblem for interfacing with PCD PC
    using fieldsplit backend. Convection fieldsplit submatrices
    are extracted as shallow or deep submatrices according to
    ``deep_submats`` parameter."""
    def __init__(self, is_u, is_p, deep_submats=False):
        # Store isets
        self.is_u = is_u
        self.is_p = is_p

        # Choose submatrix implementation
        assert isinstance(deep_submats, bool)
        if deep_submats:
            self.assemble_operator = self._assemble_operator_deep
        else:
            self.assemble_operator = self._assemble_operator_shallow
            self._scratch = {}


    # FIXME: We are mutating input pcd_problem
    @classmethod
    def reclass(cls, pcd_problem, is_u, is_p, deep_submats=False):
        """Reclasses instance of PCDProblem into this class"""
        if isinstance(pcd_problem, cls):
            raise TypeError("Cannot reclass. Already this class.")
        pcd_problem.__class__ = cls
        cls.__init__(pcd_problem, is_u, is_p, deep_submats=deep_submats)


    def setup_ksp_Ap(self, ksp):
        """Setup pressure Laplacian ksp and assemble matrix"""
        self.setup_ksp_once(ksp, self.ap, self.is_p)


    def setup_ksp_Mp(self, ksp):
        """Setup pressure mass matrix ksp and assemble matrix"""
        self.setup_ksp_once(ksp, self.mp, self.is_p)


    def setup_ksp_Mu(self, ksp):
        """Setup velocity mass matrix ksp and assemble matrix"""
        self.setup_ksp_once(ksp, self.mu, self.is_u)


    def setup_mat_Kp(self, mat=None):
        """Setup and assemble pressure convection
        matrix and return it"""
        return self.assemble_operator(self.kp, self.is_p, submat=mat)


    def setup_mat_Fp(self, mat=None):
        """Setup and assemble pressure convection-diffusion
        matrix and return it"""
        return self.assemble_operator(self.fp, self.is_p, submat=mat)


    def apply_pcd_bcs(self, vec):
        """Apply bcs to intermediate pressure vector of PCD pc"""
        self.apply_bcs(vec, self.pcd_bcs, self.is_p)


    def setup_ksp_once(self, ksp, assembler_func, iset):
        """Assemble into operator of given ksp if not yet assembled"""
        mat = ksp.getOperators()[0]
        # FIXME: This logic that it is created once should be visible
        #        in higher level, not in these internals
        # FIXME: Shouldn't we check mat.isAssembled()
        if mat.type is None:
            # FIXME: Could have shared work matrix
            A = dolfin.PETScMatrix(mat.comm)
            assembler_func(A)
            mat = self._get_deep_submat(A, iset, submat=None)
            mat.setOption(PETSc.Mat.Option.SPD, True)
            ksp.setOperators(mat, mat)
            assert ksp.getOperators()[0].type is not None


    def _assemble_operator_shallow(self, assemble_func, iset, submat=None):
        """Assemble operator of given name using shallow submat"""
        # Allocate dolfin matrix and store it for future
        dolfin_mat = self._scratch.get(assemble_func, None)
        if dolfin_mat is None:
            self._scratch[assemble_func] = \
                    dolfin_mat = dolfin.PETScMatrix(self.mpi_comm())

        # Assemble dolfin matrix everytime
        assemble_func(dolfin_mat)

        # FIXME: This logic that it is created once should be visible
        #        in higher level, not in these internals
        # Create shallow submatrix once
        if submat is None or submat.type is None:
            submat = self._get_shallow_submat(dolfin_mat, iset, submat=submat)
            assert submat.type is not None

        return submat


    def _assemble_operator_deep(self, assemble_func, iset, submat=None):
        """Assemble operator of given name using shallow submat"""
        # FIXME: Could have shared work matrix
        dolfin_mat = dolfin.PETScMatrix(self.mpi_comm())
        assemble_func(dolfin_mat)
        return self._get_deep_submat(dolfin_mat, iset, submat=submat)


    def apply_bcs(self, vec, bcs_getter, iset):
        """Transform dolfin bcs obtained using ``bcs_getter`` function
        into fieldsplit subBCs and apply them to fieldsplit vector.
        SubBCs are cached."""
        # Fetch subbcs from cache or construct it
        subbcs = getattr(self, "_subbcs", None)
        if subbcs is None:
            bcs = bcs_getter()
            bcs = [bcs] if isinstance(bcs, dolfin.DirichletBC) else bcs
            subbcs = [SubfieldBC(bc, iset) for bc in bcs]
            self._subbcs = subbcs

        # Apply bcs
        for bc in subbcs:
            bc.apply(vec)


    @staticmethod
    def _get_deep_submat(dolfin_mat, iset, submat=None):
        return dolfin_mat.mat().getSubMatrix(iset, iset, submat=submat)


    @staticmethod
    def _get_shallow_submat(dolfin_mat, iset, submat=None):
        if submat is None:
            submat = PETSc.Mat().create(iset.comm)
        return submat.createSubMatrix(dolfin_mat.mat(), iset, iset)
