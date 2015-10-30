# Copyright (C) 2014-2015 Jan Blechta and Martin Rehor
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

from dolfin import (PETScKrylovSolver, compile_extension_module,
                    as_backend_type, PETScMatrix, PETScVector,
                    SystemAssembler, assemble)
from petsc4py import PETSc

__all__ = ['SimpleFieldSplitSolver', 'PCDFieldSplitSolver']

class SimpleFieldSplitSolver(PETScKrylovSolver):
    """This class implements fieldsplit preconditioner for "Stokes-like" problems."""

    def __init__(self, space, ksptype):
        """Arguments:
             space   ... instance of dolfin's MixedFunctionSpace
             ksptype ... krylov solver type, see help(PETSc.KSP.Type)
        """
        # Setup KSP
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType(ksptype)

        # Setup FIELDSPLIT preconditioning
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
        is0 = dofmap_dofs_is(space.sub(0).dofmap())
        is1 = dofmap_dofs_is(space.sub(1).dofmap())
        pc.setFieldSplitIS(['u', is0], ['p', is1])
        is0.destroy()
        is1.destroy()

        # Create helper class to set options for external packages (e.g. HYPRE)
        self._opts = PETSc.Options()

        # Init mother class
        PETScKrylovSolver.__init__(self, ksp)

        # Set up ksp and pc
        self.default_settings()

    def opts(self):
        """Returns helper `PETSc.Options` class, so that command-line options
        can be set using either DOLFIN's `PETScOptions.set("some_option", value)`
        or `solver.opts().setValue("-some_option", value)`, where solver is an
        instance of the `SimpleFieldSplitSolver` class."""
        return self._opts

    def default_settings(self):
        """Default settings for the solver. This method is first called
        by the constructor."""
        # Get KSP and PC contexts
        ksp = self.ksp() # ksp is obtained from DOLFIN's PETScKrylovSolver
        pc = ksp.getPC()

        # Get sub-KSPs, sub-PCs
        ksp0, ksp1 = pc.getFieldSplitSubKSP()
        pc0, pc1 = ksp0.getPC(), ksp1.getPC()

        # Setup approximation of 00-block inverse (Hypre AMG)
        ksp0.setType(PETSc.KSP.Type.RICHARDSON)
        ksp0.max_it = 1
        pc0.setType(PETSc.PC.Type.HYPRE)
        self._opts.setValue("-fieldsplit_u_pc_hypre_type", "boomeramg")
        #self._opts.setValue("-fieldsplit_u_pc_hypre_boomeramg_cycle_type", "W")

        # Setup approximation of 11-block inverse (Chebyshev semi-iteration)
        ksp1.setType(PETSc.KSP.Type.CHEBYSHEV)
        ksp1.max_it = 5
        pc1.setType(PETSc.PC.Type.JACOBI)

    def setFromOptions(self):
        """Do the set up from command-line options."""
        # Get KSP and PC contexts
        ksp = self.ksp() # ksp is obtained from DOLFIN's PETScKrylovSolver
        pc = ksp.getPC()
        # Get sub-KSPs, sub-PCs
        ksp0, ksp1 = pc.getFieldSplitSubKSP()
        pc0, pc1 = ksp0.getPC(), ksp1.getPC()
        # Call setFromOptions method for all KSPs and PCs
        ksp0.setFromOptions()
        pc0.setFromOptions()
        ksp1.setFromOptions()
        pc1.setFromOptions()
        ksp.setFromOptions()
        pc.setFromOptions()

class PCDFieldSplitSolver(PETScKrylovSolver):
    """This class implements PCD preconditioner for Navier-Stokes problems."""

    def __init__(self, space, ksptype='gmres', insolver='lu'):
        """Arguments:
             space    ... instance of dolfin's MixedFunctionSpace
             ksptype  ... krylov solver type, see help(PETSc.KSP.Type)
             insolver ... direct/iterative inner solver (choices: 'lu', 'it')
        """
        # Setup GMRES with RIGHT preconditioning
        ksptype_petsc_name = "bcgs" if ksptype == "bicgstab" else ksptype
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType(ksptype_petsc_name)
        ksp.setPCSide(PETSc.PC.Side.RIGHT)

        # Setup SCHUR with UPPER factorization
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)
        is0 = dofmap_dofs_is(space.sub(0).dofmap())
        is1 = dofmap_dofs_is(space.sub(1).dofmap())
        pc.setFieldSplitIS(['u', is0], ['p', is1])

        # Store what needed
        self._is0 = is0
        self._is1 = is1
        self._insolver = insolver

        # Init mother class
        PETScKrylovSolver.__init__(self, ksp)

    def setup(self, *args):
        # Setup KSP and PC
        ksp = self.ksp() # ksp is obtained from DOLFIN's PETScKrylovSolver
        ksp.setUp()
        pc = ksp.getPC()
        pc.setUp()

        # Get sub-KSPs, sub-PCs
        ksp0, ksp1 = pc.getFieldSplitSubKSP()
        pc0, pc1 = ksp0.getPC(), ksp1.getPC()

        # Setup approximation of 00-block inverse
        ksp0.setFromOptions()
        ksp0.setUp()
        pc0.setFromOptions()
        pc0.setUp()

        # Setup approximation of Schur complement (11-block) inverse
        ksp1.setType(PETSc.KSP.Type.PREONLY)
        ksp1.setUp()
        pc1.setType(PETSc.PC.Type.PYTHON)
        pc1.setPythonContext(PCDctx(self._insolver, self.ksp(),
                                    self._is0, self._is1, *args))
        pc1.setUp()


class PCDctx(object):
    """Python context for PCD preconditioner."""

    def __init__(self, *args):
        if args:
            self._ctxapply = 0
            self._opts = PETSc.Options()
            self.set_operators(*args)
            self.assemble_operators()

    def set_operators(self, insolver, ksp, isu, isp,
                      mu, mp, fp, ap, Lp, bcs_Ap, strategy):
        """Collects an index set to identify block corresponding to Schur
        complement in the system matrix, variational forms and boundary
        conditions to assemble corresponding matrix operators."""
        self._ksp = ksp
        self._isu = isu
        self._isp = isp
        self._mu = mu    # -> velocity mass matrux Mu
        self._mp = mp    # -> pressure mass matrix Mp
        self._ap = ap    # -> pressure Laplacian Ap
        self._fp = fp    # -> pressure convection-diffusion Fp
        self._Lp = Lp    # -> dummy right hand side vector (not used)
        # TODO: What are correct BCs?
        self._bcs_Mp = [] if strategy == 'B' else bcs_Ap
        self._bcs_Ap = bcs_Ap
        self._bcs_Fp = bcs_Ap
        self._insolver = insolver
        self._strategy = strategy

    def assemble_operators(self):
        """Prepares operators for PCD preconditioning."""
        #
        # TODO: Some operators can be assembled only once outside PCDctx,
        #       e.g. velocity mass matrix. (In the current naive approach we
        #       assemble those operators in each nonlinear iteration.)
        #
        # TODO: Can we use SystemAssembler to assemble operators in a symmetric
        #       way whithout simultaneous assembly of rhs vector? (Supposing
        #       that we are dealing only with homogeneous bcs.)
        #
        # Assemble Ap
        if self._strategy in ['A', 'B']:
            self._Ap = PETScMatrix()
            assemble(self._ap, tensor=self._Ap)
            for bc in self._bcs_Ap:
                bc.apply(self._Ap)
            self._Ap = self._Ap.mat().getSubMatrix(self._isp, self._isp)
        else:
            self._Ap = self._assemble_Ap_approx(multiply='Bt')
        # Assemble Fp
        self._Fp = PETScMatrix()
        assemble(self._fp, tensor=self._Fp)
        for bc in self._bcs_Fp:
            bc.apply(self._Fp)
        self._Fp = self._Fp.mat().getSubMatrix(self._isp, self._isp)
        # Assemble Mp
        self._Mp = PETScMatrix()
        assemble(self._mp, tensor=self._Mp)
        for bc in self._bcs_Mp:
            bc.apply(self._Mp)
        self._Mp = self._Mp.mat().getSubMatrix(self._isp, self._isp)
        self.prepare_factors()

    def _assemble_Ap_approx(self, multiply='Bt'):
        """Assembly of an approximation of the pressure Laplacian that is more
        appropriate for nonstationary problems, i.e. Ap = B*diag(Mu)^{-1}*B^T,
        where Mu is the velocity mass matrix. (Note that for inflow-outflow
        problems this operator is nonsingular and we do not need to prescribe
        any artificial boundary conditions for pressure. For enclosed flows
        one can solve the system using iterative solvers.)"""
        # Assemble velocity mass matrix
        # NOTE: There is no need to apply Dirichlet bcs for velocity on the
        #       velocity mass matrix as long as those bsc have been applied on
        #       the system matrix A. Thus, Bt contains zero rows corresponding
        #       to test functions on the Dirichlet boundary.
        Mu = PETScMatrix()
        assemble(self._mu, tensor=Mu)
        Mu = Mu.mat().getSubMatrix(self._isu, self._isu)
        # Get diagonal of the velocity mass matrix
        iT = Mu.getDiagonal()
        # Make inverse of diag(Mu)
        iT.reciprocal()
        # Make square root of the diagonal and use it for scaling
        iT.sqrtabs()
        # Extract 01-block, i.e. "grad", from the matrix operator A
        A, P = self._ksp.getOperators()
        if multiply != 'B':
            Bt = A.getSubMatrix(self._isu, self._isp)
            Bt.diagonalScale(L=iT)
        # Extract 10-block, i.e. "-div", from the matrix operator A
        if multiply != 'Bt':
            B = A.getSubMatrix(self._isp, self._isu)
            B.diagonalScale(R=iT)
        # Prepare Ap
        if multiply == 'Bt':
            # Correct way, i.e. Ap = Bt^T*Bt
            Ap = Bt.transposeMatMult(Bt)
        elif multiply == 'B':
            # Wrong way, i.e. Ap = B*B^T (B^T doesn't contain zero rows as Bt)
            Ap = B.matTransposeMult(B)
            #Ap = PETSc.Mat().createNormal(B)
        else:
            # Explicit multiplication, i.e. Ap = B*Bt
            Ap = B.matMult(Bt)
        return Ap

    def prepare_factors(self):
        # Prepare Mp factorization
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        pc = ksp.getPC()
        if self._insolver == 'lu':
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc.setType(PETSc.PC.Type.LU)
            #pc.setFactorSolverPackage('umfpack')
        else:
            #ksp.setType(PETSc.KSP.Type.CG)
            #pc.setType(PETSc.PC.Type.HYPRE)
            ksp.setType(PETSc.KSP.Type.CHEBYSHEV)
            ksp.max_it = 5
            pc.setType(PETSc.PC.Type.JACOBI)
            # FIXME: The following estimates are valid only for continuous pressure.
            self._opts.setValue("-ksp_chebyshev_eigenvalues", "0.5, 2.0")
        #self._Mp.setOption(PETSc.Mat.Option.SPD, True)
        ksp.setOperators(self._Mp)
        ksp.setFromOptions()
        ksp.setUp()
        pc.setUp()
        self._ksp_Mp = ksp

        # Prepare Ap factorization
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        pc = ksp.getPC()
        if self._insolver == 'lu':
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc.setType(PETSc.PC.Type.LU)
            #pc.setFactorSolverPackage('umfpack')
        else:
            ksp.setType(PETSc.KSP.Type.RICHARDSON)
            ksp.max_it = 2
            pc.setType(PETSc.PC.Type.HYPRE)
            #self._opts.setValue("-pc_hypre_type", "boomeramg") # this is default
            #self._opts.setValue("-pc_hypre_boomeramg_cycle_type", "W")
        #self._Ap.setOption(PETSc.Mat.Option.SPD, True)
        ksp.setOperators(self._Ap)
        ksp.setFromOptions()
        ksp.setUp()
        pc.setUp()
        self._ksp_Ap = ksp

    def apply(self, pc, x, y):
        """This method implements the following action (x ... input vector,
        y ... output vector), cf. PCShellSetApply:
            $y = S^{-1} x = -A_p^{-1} F_p M_p^{-1} x$,
        where $S = -M_p F_p^{-1} A_p$ approximates Schur complement $-B F^{-1} B^{T}$."""
        # TODO: Try matrix-free!
        # TODO: Is modification of x safe?
        self._ctxapply += 1
        # if self._ctxapply == 1:
        #     x.view()
        if self._strategy in ['A', 'C']:
            self._ksp_Mp.solve(x, y) # y = M_p^{-1} x
            # NOTE: Preconditioning with sole M_p in place of 11-block works for low Re.
            self._Fp.mult(-y, x) # x = -F_p y
            # TRY: Apply boundary conditions to x. (There must be some smarter
            #      way how to do that. Nevertheless, it seems that direct
            #      modification of RHS does not lead to anticipated results.)
            # lgmap = PETSc.LGMap().createIS(self._isp) # local to global mapping
            # x.setLGMap(lgmap)
            # lgmap = lgmap.indices.tolist() # FIXME: brute force attack
            # indices = []
            # for bc in self._bcs_Ap:
            #     bc_map = bc.get_boundary_values()
            #     for key in bc_map.keys():
            #         indices.append(lgmap.index(key)) # FIXME: brute force attack
            #     x.setValues(indices, bc_map.values())
            # if self._ctxapply == 1:
            #     print indices
            #     x.view()
            self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
        else:
            # Interchange the order of operators for strategy B
            # TRY: Apply boundary conditions to x. (There must be some smarter
            #      way how to do that. Nevertheless, it seems that direct
            #      modification of RHS does not lead to anticipated results.)
            # lgmap = PETSc.LGMap().createIS(self._isp) # local to global mapping
            # x.setLGMap(lgmap)
            # lgmap = lgmap.indices.tolist() # FIXME: brute force attack
            # indices = []
            # for bc in self._bcs_Ap:
            #     bc_map = bc.get_boundary_values()
            #     for key in bc_map.keys():
            #         indices.append(lgmap.index(key)) # FIXME: brute force attack
            #     x.setValues(indices, bc_map.values())
            # # if self._ctxapply == 1:
            # #     print indices
            # #     x.view()
            self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
            # if self._ctxapply == 1:
            #     y.view()
            self._Fp.mult(-y, x) # x = -F_p y
            self._ksp_Mp.solve(x, y) # y = M_p^{-1} x


dofmap_dofs_is_cpp_code = """
#ifdef SWIG
%include "petsc4py/petsc4py.i"
#endif

#include <vector>
#include <petscis.h>
#include <dolfin/fem/GenericDofMap.h>

namespace dolfin {

  IS dofmap_dofs_is(const GenericDofMap& dofmap)
  {
    const std::vector<dolfin::la_index> dofs = dofmap.dofs();
    IS is;
    ISCreateGeneral(PETSC_COMM_WORLD, dofs.size(), dofs.data(),
                    PETSC_COPY_VALUES, &is);
    return is;
  }

}
"""

dofmap_dofs_is = \
    compile_extension_module(dofmap_dofs_is_cpp_code).dofmap_dofs_is
del dofmap_dofs_is_cpp_code
