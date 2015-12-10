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

import dolfin
from petsc4py import PETSc

__all__ = ['PCDPC_ESW']

class BasePCDPC(object):
    """Base python context for pressure convection diffusion (PCD)
    preconditioners."""

    def __init__(self):
        self._ksp_Ap = self._prepare_Ap_fact()
        self._ksp_Mp = self._prepare_Mp_fact()

    def apply(self, pc, x, y):
        raise NotImplementedError

    def custom_setup(self, *args, **kwargs):
        pass

    def _isset_error(self, kwarg):
        dolfin.error("Keyword argument '%s' has not been set." % kwarg)

    def _prepare_Ap_fact(self):
        # Prepare Ap factorization
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        # Default settings
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        # Update settings
        ksp.setOptionsPrefix("fieldsplit_p_PCD_Ap_")
        ksp.setFromOptions()
        # Return created ksp
        return ksp

    def _prepare_Mp_fact(self):
        # Prepare Mp factorization
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        # Default settings
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        # Update settings
        ksp.setOptionsPrefix("fieldsplit_p_PCD_Mp_")
        ksp.setFromOptions()
        # Return created ksp
        return ksp

class PCDPC_ESW(BasePCDPC):
    """This class implements PCD preconditioning proposed by Elman, Silvester
    and Wathen (2014)."""

    def __init__(self):
        BasePCDPC.__init__(self)
        self._isset_Ap = False # pressure laplacian
        self._isset_Fp = False # pressure convection-difusion
        self._isset_Mp = False # pressure mass matrix
        self._bc_indices = None # indices corresponding to outflow boundary

    def apply(self, pc, x, y):
        """This method implements the action of the inverse of the approximate
        Schur complement $\hat{S} = -M_p F_p^{-1} A_p$, that is

            $y = \hat{S}^{-1} x = -A_p^{-1} F_p M_p^{-1} x$.
        """
        timer = dolfin.Timer("FENaPack: call PCDPC_ESW.apply")
        timer.start()
        self._ksp_Mp.solve(x, y) # y = M_p^{-1} x
        self._Fp.mult(-y, x)     # x = -F_p y
        self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
        timer.stop()
        # TODO: Try matrix-free!
        # TODO: Is modification of x safe?

    def custom_setup(self, is0, is1, Ap=None, Fp=None, Mp=None, bcs=None):
        timer = dolfin.Timer("FENaPack: call PCDPC_ESW.custom_setup")
        timer.start()
        # Update bcs
        if bcs:
            # Make sure that 'bcs' is a list
            if not isinstance(bcs, list):
                bcs = [bcs]
            # Get indices
            indices = []
            for bc in bcs:
                indices += bc.get_boundary_values().keys()
            # Remove redundant indices and save the result
            self._bc_indices = list(set(indices))
        elif not self._bc_indices:
            self._isset_error("bcs")
        # Update Ap
        if Ap:
            # Get PETSc Mat object
            Ap = dolfin.as_backend_type(Ap).mat()
            # Prepare scaling factor for application of bcs
            scaling_factor = Ap.getDiagonal() # get long vector
            scaling_factor.set(1.0)
            scaling_factor.setValuesLocal(self._bc_indices,
                                          len(self._bc_indices)*[2.0,])
            scaling_factor = scaling_factor.getSubVector(is1) # short vector
            # Get submatrix corresponding to 11-block of the problem matrix
            Ap = Ap.getSubMatrix(is1, is1)
            # Apply Dirichlet conditions using a ghost layer of elements
            # outside the outflow boundary (mimic finite differences,
            # i.e. augment diagonal with values from "ghost elements")
            diag = Ap.getDiagonal()
            diag.pointwiseMult(diag, scaling_factor)
            Ap.setDiagonal(diag)
            # Store matrix in the corresponding ksp
            self._ksp_Ap.setOperators(Ap)
            # Update flag
            self._isset_Ap = True
            # TODO: Think about explicit destruction of PETSc objects.
        elif not self._isset_Ap:
            self._isset_error("Ap")
        # Update Fp
        if Fp:
            # Get PETSc Mat object
            Fp = dolfin.as_backend_type(Fp).mat()
            # Prepare scaling factor for application of bcs
            scaling_factor = Fp.getDiagonal() # get long vector
            scaling_factor.set(1.0)
            scaling_factor.setValuesLocal(self._bc_indices,
                                          len(self._bc_indices)*[2.0,])
            scaling_factor = scaling_factor.getSubVector(is1) # short vector
            # Get submatrix corresponding to 11-block of the problem matrix
            Fp = Fp.getSubMatrix(is1, is1)
            # Apply Dirichlet conditions using a ghost layer of elements
            # outside the outflow boundary (mimic finite differences,
            # i.e. augment diagonal with values from "ghost elements")
            diag = Fp.getDiagonal()
            diag.pointwiseMult(diag, scaling_factor)
            Fp.setDiagonal(diag)
            # Store matrix
            self._Fp = Fp
            # Update flag
            self._isset_Fp = True
        elif not self._isset_Fp:
            self._isset_error("Fp")
        # Update Mp
        if Mp:
            Mp = dolfin.as_backend_type(Mp).mat()
            Mp = Mp.getSubMatrix(is1, is1)
            # Store matrix in the corresponding ksp
            self._ksp_Mp.setOperators(Mp)
            # Update flag
            self._isset_Mp = True
        elif not self._isset_Mp:
            self._isset_error("Mp")
        timer.stop()

# -----------------------------------------------------------------------------
# TODO: Re-implement PCDctx following the new template provided above.

# class PCDctx(object):
#     """Python context for PCD preconditioner."""

#     def __init__(self, *args):
#         if args:
#             self._opts = PETSc.Options()
#             self.set_operators(*args)
#             self.assemble_operators()
#             self.prepare_factors()

#     def set_operators(self, insolver, ksp, isu, isp, strategy, flag_BQBt,
#                       mu, mp, ap, kp, fp, Lp, bcs_pcd, nu):
#         """Collects an index set to identify block corresponding to Schur
#         complement in the system matrix, variational forms and boundary
#         conditions to assemble corresponding matrix operators."""
#         self._ksp = ksp
#         self._isu = isu
#         self._isp = isp
#         self._mu = mu    # -> velocity mass matrux Mu
#         self._mp = mp    # -> pressure mass matrix Mp
#         self._ap = ap    # -> pressure Laplacian Ap
#         self._kp = kp    # -> pressure convection Kp
#         self._fp = fp    # -> pressure convection-diffusion Fp
#         self._Lp = Lp    # -> dummy right hand side vector
#         self._bcs_pcd = bcs_pcd
#         self._insolver = insolver
#         self._strategy = strategy
#         self._flag_BQBt = flag_BQBt
#         self._nu = nu

#     def assemble_operators(self):
#         """Prepares operators for PCD preconditioning."""
#         #
#         # TODO: Some operators can be assembled only once outside PCDctx,
#         #       e.g. velocity mass matrix. (In the current naive approach we
#         #       assemble those operators in each nonlinear iteration.)
#         #
#         # TODO: Can we use SystemAssembler to assemble operators in a symmetric
#         #       way whithout simultaneous assembly of rhs vector? (Supposing
#         #       that we are dealing only with homogeneous bcs.)
#         #
#         if self._strategy == 'ESW14':
#             # Apply Dirichlet conditions for Ap and Fp using a ghost layer of
#             # elements outside the outflow boundary (mimic finite differences).
#             indices = []
#             for bc in self._bcs_pcd:
#                 bc_map = bc.get_boundary_values()
#                 indices += bc_map.keys()
#             scaling_factor = PETScVector()
#             assemble(self._Lp, tensor=scaling_factor)
#             scaling_factor = scaling_factor.vec()
#             scaling_factor.set(1.0)
#             scaling_factor.setValuesLocal(indices, len(indices)*[2.0,])
#             scaling_factor = scaling_factor.getSubVector(self._isp)
#             # Assemble Ap
#             if self._flag_BQBt:
#                 # Ap = B*T^{-1}*Bt (appropriate for nonstationary problems)
#                 self._Ap = self._assemble_BQBt(multiply='Bt')
#             else:
#                 # Ap assembled from UFL form
#                 self._Ap = PETScMatrix()
#                 assemble(self._ap, tensor=self._Ap)
#                 self._Ap = self._Ap.mat().getSubMatrix(self._isp, self._isp)
#                 # Augment diagonal with values from "ghost elements"
#                 d = self._Ap.getDiagonal()
#                 d.pointwiseMult(d, scaling_factor)
#                 self._Ap.setDiagonal(d)
#                 d.destroy()
#                 del d
#             # Assemble Fp
#             self._Fp = PETScMatrix()
#             assemble(self._fp, tensor=self._Fp)
#             self._Fp = self._Fp.mat().getSubMatrix(self._isp, self._isp)
#             # Augment diagonal with values from "ghost elements"
#             d = self._Fp.getDiagonal()
#             d.pointwiseMult(d, scaling_factor)
#             self._Fp.setDiagonal(d)
#             d.destroy()
#             del d
#             # Assemble Mp
#             self._Mp = PETScMatrix()
#             assemble(self._mp, tensor=self._Mp)
#             self._Mp = self._Mp.mat().getSubMatrix(self._isp, self._isp)
#         elif self._strategy == 'BMR15':
#             # Assemble Ap
#             if self._flag_BQBt:
#                 # Ap = B*diag(Mu)^{-1}*Bt (appropriate for nonstationary problems)
#                 self._Ap = self._assemble_BQBt(multiply='Bt')
#             else:
#                 # Ap assembled from UFL form
#                 self._Ap = PETScMatrix()
#                 assemble(self._ap, tensor=self._Ap)
#                 for bc in self._bcs_pcd:
#                     bc.apply(self._Ap)
#                 self._Ap = self._Ap.mat().getSubMatrix(self._isp, self._isp)
#             # Assemble Kp
#             self._Kp = PETScMatrix()
#             assemble(self._kp, tensor=self._Kp)
#             self._Kp = self._Kp.mat().getSubMatrix(self._isp, self._isp)
#             # Assemble Mp
#             self._Mp = PETScMatrix()
#             assemble(self._mp, tensor=self._Mp)
#             self._Mp = self._Mp.mat().getSubMatrix(self._isp, self._isp)
#         else:
#             error("Unknown PCD strategy.")

#     def _assemble_BQBt(self, multiply='Bt'):
#         """Assembly of an approximation of the pressure Laplacian that is more
#         appropriate for nonstationary problems, i.e. Ap = B*diag(Mu)^{-1}*B^T,
#         where Mu is the velocity mass matrix. (Note that for inflow-outflow
#         problems this operator is nonsingular and we do not need to prescribe
#         any artificial boundary conditions for pressure. For enclosed flows
#         one can solve the system using iterative solvers.)"""
#         # Assemble velocity mass matrix
#         # NOTE: There is no need to apply Dirichlet bcs for velocity on the
#         #       velocity mass matrix as long as those bsc have been applied on
#         #       the system matrix A. Thus, Bt contains zero rows corresponding
#         #       to test functions on the Dirichlet boundary.
#         Mu = PETScMatrix()
#         assemble(self._mu, tensor=Mu)
#         Mu = Mu.mat().getSubMatrix(self._isu, self._isu)
#         # Get diagonal of the velocity mass matrix
#         iT = Mu.getDiagonal()
#         # Make inverse of diag(Mu)
#         iT.reciprocal()
#         # Make square root of the diagonal and use it for scaling
#         iT.sqrtabs()
#         # Extract 01-block, i.e. "grad", from the matrix operator A
#         A, P = self._ksp.getOperators()
#         if multiply != 'B':
#             Bt = A.getSubMatrix(self._isu, self._isp)
#             Bt.diagonalScale(L=iT)
#         # Extract 10-block, i.e. "-div", from the matrix operator A
#         if multiply != 'Bt':
#             B = A.getSubMatrix(self._isp, self._isu)
#             B.diagonalScale(R=iT)
#         # Prepare Ap
#         if multiply == 'Bt':
#             # Correct way, i.e. Ap = Bt^T*Bt
#             Ap = Bt.transposeMatMult(Bt)
#         elif multiply == 'B':
#             # Wrong way, i.e. Ap = B*B^T (B^T doesn't contain zero rows as Bt)
#             Ap = B.matTransposeMult(B)
#             #Ap = PETSc.Mat().createNormal(B)
#         else:
#             # Explicit multiplication, i.e. Ap = B*Bt
#             Ap = B.matMult(Bt)
#         return Ap

#     def prepare_factors(self):
#         # Prepare Mp factorization
#         ksp = PETSc.KSP()
#         ksp.create(PETSc.COMM_WORLD)
#         pc = ksp.getPC()
#         if self._insolver == 'lu':
#             ksp.setType(PETSc.KSP.Type.PREONLY)
#             pc.setType(PETSc.PC.Type.LU)
#             #pc.setFactorSolverPackage('umfpack')
#         else:
#             #ksp.setType(PETSc.KSP.Type.CG)
#             #pc.setType(PETSc.PC.Type.HYPRE)
#             ksp.setType(PETSc.KSP.Type.CHEBYSHEV)
#             ksp.max_it = 5
#             pc.setType(PETSc.PC.Type.JACOBI)
#             # FIXME: The following estimates are valid only for continuous pressure.
#             self._opts.setValue("-ksp_chebyshev_eigenvalues", "0.5, 2.0")
#         #self._Mp.setOption(PETSc.Mat.Option.SPD, True)
#         ksp.setOperators(self._Mp)
#         ksp.setFromOptions()
#         ksp.setUp()
#         pc.setUp()
#         self._ksp_Mp = ksp

#         # Prepare Ap factorization
#         ksp = PETSc.KSP()
#         ksp.create(PETSc.COMM_WORLD)
#         pc = ksp.getPC()
#         if self._insolver == 'lu':
#             ksp.setType(PETSc.KSP.Type.PREONLY)
#             pc.setType(PETSc.PC.Type.LU)
#             #pc.setFactorSolverPackage('umfpack')
#         else:
#             ksp.setType(PETSc.KSP.Type.RICHARDSON)
#             ksp.max_it = 2
#             pc.setType(PETSc.PC.Type.HYPRE)
#             #self._opts.setValue("-pc_hypre_type", "boomeramg") # this is default
#             #self._opts.setValue("-pc_hypre_boomeramg_cycle_type", "W")
#         #self._Ap.setOption(PETSc.Mat.Option.SPD, True)
#         ksp.setOperators(self._Ap)
#         ksp.setFromOptions()
#         ksp.setUp()
#         pc.setUp()
#         self._ksp_Ap = ksp

#     def apply(self, pc, x, y):
#         """This method implements the action of the inverse of approximate
#         Schur complement S, cf. PCShellSetApply."""
#         # TODO: Try matrix-free!
#         # TODO: Is modification of x safe?
#         if self._strategy == 'ESW14':
#             # $y = S^{-1} x = -A_p^{-1} F_p M_p^{-1} x$,
#             # where $S = -M_p F_p^{-1} A_p$ approximates $-B F^{-1} B^{T}$.
#             self._ksp_Mp.solve(x, y) # y = M_p^{-1} x
#             # NOTE: Preconditioning with sole M_p in place of 11-block works for low Re.
#             self._Fp.mult(-y, x) # x = -F_p y
#             self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
#         elif self._strategy == 'BMR15':
#             # $y = S^{-1} x = -M_p^{-1} (\nu x + K_p A_p^{-1} x)}$,
#             # where $S = -A_p F_p^{-1} M_p = -A_p (\nu A_p + K_p)^{-1} M_p$
#             # approximates $-B F^{-1} B^{T}$.
#             # FIXME: This is an inefficient approximation
#             x0 = x.copy()
#             lgmap = PETSc.LGMap().createIS(self._isp) # local to global mapping
#             x.setLGMap(lgmap)
#             lgmap = lgmap.indices.tolist()
#             indices = []
#             for bc in self._bcs_pcd:
#                 bc_map = bc.get_boundary_values()
#                 for key in bc_map.keys():
#                     indices.append(lgmap.index(key))  # FIXME: brute force attack
#                 if not self._flag_BQBt:
#                     x.setValues(indices, bc_map.values()) # apply bcs to rhs
#             self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
#             self._Kp.mult(-y, x)
#             x.axpy(-self._nu, x0)    # x = -K_p y - nu*x0
#             self._ksp_Mp.solve(x, y) # y = M_p^{-1} x
#         else:
#             error("Unknown PCD strategy.")
