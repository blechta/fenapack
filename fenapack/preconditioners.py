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

from fenapack.field_split_utils import SubfieldBC

__all__ = ['PCDPC_ESW', 'UnsteadyPCDPC_ESW', 'PCDPC_BMR']

class BasePCDPC(object):
    """Base python context for pressure convection diffusion (PCD)
    preconditioners."""

    def __init__(self):
        self._ksp_Ap = self._prepare_Ap_fact()
        self._ksp_Mp = self._prepare_Mp_fact()

    def apply(self, pc, x, y):
        raise NotImplementedError

    def setFromOptions(self, pc):
        options_prefix = pc.getOptionsPrefix()
        # Update Ap
        self._ksp_Ap.setOptionsPrefix(options_prefix+"PCD_Ap_")
        self._ksp_Ap.setFromOptions()
        # Update Mp
        self._ksp_Mp.setOptionsPrefix(options_prefix+"PCD_Mp_")
        self._ksp_Mp.setFromOptions()

    def set_operators(self, is0, is1, A, P, **shur_approx):
        """The index sets is0, is1 define particular blocks in a field-splitted
        matrix.

        From A and P one can extract their particular blocks and use them to
        build the approximate Schur complement.

        **Overloaded versions**

          Other components used to build the approximate Schur complement
          matrix can be provided as optional keyword arguments. These
          components differ depending on the strategy used for PCD
          preconditioning.
        """
        pass

    def _raise_attribute_error(self, attr):
        raise AttributeError(
            "Attribute '_%s' must be set using '%s.set_operators()'"
            % (attr, type(self).__name__))

    def _prepare_Ap_fact(self):
        # Prepare Ap factorization
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        # Default settings
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        # Return created ksp
        return ksp

    def _prepare_Mp_fact(self):
        # Prepare Mp factorization
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        # Default settings
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        # Return created ksp
        return ksp

class PCDPC_ESW(BasePCDPC):
    """This class implements PCD preconditioning proposed by Elman, Silvester
    and Wathen (2014)."""

    def apply(self, pc, x, y):
        """This method implements the action of the inverse of the approximate
        Schur complement $\hat{S} = -M_p F_p^{-1} A_p$, that is

            $y = \hat{S}^{-1} x = -A_p^{-1} F_p M_p^{-1} x$.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_ESW apply")
        timer.start()
        self._ksp_Mp.solve(x, y) # y = M_p^{-1} x
        self._Fp.mult(-y, x)     # x = -F_p y
        self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
        timer.stop()
        # TODO: Try matrix-free!
        # TODO: Is modification of x safe?

    def set_operators(self, is0, is1, A, P,
                      Mp=None, Ap=None, Fp=None, bcs=None):
        """Set operators for the approximate Schur complement matrix

        *Arguments*
            is0, is1 (`petsc4py.PETSc.IS`)
                The index sets defining blocks in the field-splitted matrix.
            A (:py:class:`GenericMatrix`)
                The system matrix. [NOT USED]
            P (:py:class:`GenericMatrix`)
                The preconditioning matrix. [NOT USED]

        *Keyword arguments*
            Mp (:py:class:`GenericMatrix`)
                The matrix containing pressure mass matrix as its 11-block.
            Ap (:py:class:`GenericMatrix`)
                The matrix containing pressure laplacian as its 11-block.
            Fp (:py:class:`GenericMatrix`)
                The matrix containing pressure convection-diffusion as its
                11-block.
            bcs (:py:class:`DirichletBC`)
                List of boundary conditions that will be applied on Ap and Fp.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_ESW set_operators")
        # Prepare bcs for adjusting field split matrix
        if bcs is not None:
            if not hasattr(bcs, "__iter__"):
                bcs = [bcs]
            self._bcs = [SubfieldBC(bc, is1) for bc in bcs]
        elif not hasattr(self, "_bcs"):
            self._raise_attribute_error("bcs")
        # Update Ap
        if Ap is not None:
            self._Ap = dolfin.as_backend_type(Ap).mat().getSubMatrix(is1, is1)
            # Apply boundary conditions along outflow boundary
            for bc in self._bcs:
                bc.apply_fdm(self._Ap)
            #self._Ap.setOption(PETSc.Mat.Option.SPD, True)
            self._ksp_Ap.setOperators(self._Ap)
        elif not hasattr(self, "_Ap"):
            self._raise_attribute_error("Ap")
        # Update Fp
        if Fp is not None:
            self._Fp = dolfin.as_backend_type(Fp).mat().getSubMatrix(is1, is1)
            # Apply boundary conditions along outflow boundary
            for bc in self._bcs:
                bc.apply_fdm(self._Fp)
        elif not hasattr(self, "_Fp"):
            self._raise_attribute_error("Fp")
        # Update Mp
        if Mp:
            self._Mp = dolfin.as_backend_type(Mp).mat().getSubMatrix(is1, is1)
            #self._Mp.setOption(PETSc.Mat.Option.SPD, True)
            self._ksp_Mp.setOperators(self._Mp)
        elif not hasattr(self, "_Mp"):
            self._raise_attribute_error("Mp")
        timer.stop()

class UnsteadyPCDPC_ESW(PCDPC_ESW):
    """This class implements PCD preconditioning proposed by Elman, Silvester
    and Wathen (2014) appropriate for unsteady problems. It derives from
    PCDPC_ESW but pressure Laplacian Ap is approximated by B*diag(Mu)^{-1}*Bt,
    where Mu is the velocity mass matrix, B corresponds to 10-block of the
    system matrix ('-div' operator) and Bt is its transpose, i.e. 01-block of
    the system matrix ('grad' operator).

    Note that Bt contains zero rows corresponding to dofs on the Dirichlet
    boundary since bcs have been applied on A. Moreover, B*diag(Mu)^{-1}*Bt is
    nonsingular for inflow-outflow problems, thus we do not need to prescribe
    any artificial boundary conditions for pressure.
    """

    def set_operators(self, is0, is1, A, P,
                      Mp=None, Mu=None, Fp=None, bcs=None):
        """Set operators for the approximate Schur complement matrix

        *Arguments*
            is0, is1 (`petsc4py.PETSc.IS`)
                The index sets defining blocks in the field splitted matrix.
            A (:py:class:`GenericMatrix`)
                The system matrix.
            P (:py:class:`GenericMatrix`)
                The preconditioning matrix. [NOT USED]

        *Keyword arguments*
            Mp (:py:class:`GenericMatrix`)
                The matrix containing pressure mass matrix as its 11-block.
            Mu (:py:class:`GenericMatrix`)
                The matrix containing velocity mass matrix as its 00-block.
            Fp (:py:class:`GenericMatrix`)
                The matrix containing pressure convection-diffusion as its
                11-block.
            bcs (:py:class:`DirichletBC`)
                List of boundary conditions that will be applied on Fp.
        """
        timer = dolfin.Timer("FENaPack: UnsteadyPCDPC_ESW set_operators")
        # Assemble Ap using A and Mu
        if Mu is not None:
            # Get velocity mass matrix as PETSc Mat object
            Mu = dolfin.as_backend_type(Mu).mat().getSubMatrix(is0, is0)
            # Get diagonal of the velocity mass matrix
            diagMu = Mu.getDiagonal()
            # Make inverse of diag(Mu)
            diagMu.reciprocal() # diag(Mu)^{-1}
            # Make square root of the diagonal and use it for scaling
            diagMu.sqrtabs() # \sqrt{diag(Mu)^{-1}}
            # Extract 01-block, i.e. "grad", from the matrix operator A
            Bt = dolfin.as_backend_type(A).mat().getSubMatrix(is0, is1)
            Bt.diagonalScale(L=diagMu) # scale rows of Bt
            # Get Ap
            self._Ap = Bt.transposeMatMult(Bt) # Ap = Bt^T*Bt
            # Set up special options
            #self._Ap.setOption(PETSc.Mat.Option.SPD, True)
            # Store matrix in the corresponding ksp
            self._ksp_Ap.setOperators(self._Ap)
        elif not hasattr(self, "_Ap"):
            self._raise_attribute_error("Ap")
        # Update remaining operators in the same way as in the parent class
        PCDPC_ESW.set_operators(self, is0, is1, A, P, Mp=Mp, Fp=Fp, bcs=bcs)
        timer.stop()

class PCDPC_BMR(BasePCDPC):
    """This class implements PCD preconditioning proposed by Blechta, Malek,
    Rehor (201?)."""

    def apply(self, pc, x, y):
        """This method implements the action of the inverse of the approximate
        Schur complement $\hat{S} = -A_p F_p^{-1} M_p$, that is

            $y = \hat{S}^{-1} x = -M_p^{-1} F_p A_p^{-1} x$.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_BMR apply")
        x0 = x.copy()
        # Apply subfield boundary conditions to rhs
        for bc in self._subfield_bcs:
            bc.apply(x)
        self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
        self._Kp.mult(-y, x)
        x.axpy(-self._nu, x0)    # x = -K_p y - nu*x0
        self._ksp_Mp.solve(x, y) # y = M_p^{-1} x
        timer.stop()
        # TODO: Try matrix-free!
        # TODO: Is modification of x safe?

    def set_operators(self, is0, is1, A, P,
                      Mp=None, Ap=None, Kp=None, nu=None, bcs=None):
        """Set operators for the approximate Schur complement matrix

        *Arguments*
            is0, is1 (`petsc4py.PETSc.IS`)
                The index sets defining blocks in the field splitted matrix.
            A (:py:class:`GenericMatrix`)
                The system matrix. [NOT USED]
            P (:py:class:`GenericMatrix`)
                The preconditioning matrix. [NOT USED]

        *Keyword arguments*
            Mp (:py:class:`GenericMatrix`)
                The matrix containing pressure mass matrix as its 11-block.
            Ap (:py:class:`GenericMatrix`)
                The matrix containing pressure laplacian as its 11-block.
            Kp (:py:class:`GenericMatrix`)
                The matrix containing pressure convection as its 11-block.
            nu (float)
                Kinematic viscosity.
            bcs (:py:class:`DirichletBC`)
                List of boundary conditions that will be applied on Ap.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_BMR set_operators")
        # Update nu
        if nu is not None:
            self._nu = nu
        elif not hasattr(self, "_nu"):
            self._raise_attribute_error("nu")
        # Prepare bcs for adjusting field split vector in PC apply
        if bcs is not None:
            if not hasattr(bcs, "__iter__"):
                bcs = [bcs]
            self._bcs = bcs
            self._subfield_bcs = [SubfieldBC(bc, is1) for bc in bcs]
        elif not hasattr(self, "_bcs"):
            self._raise_attribute_error("bcs")
        # Update Ap
        if Ap is not None:
            # Apply boundary conditions along inflow boundary
            for bc in self._bcs:
                bc.apply(Ap)
            self._Ap = dolfin.as_backend_type(Ap).mat().getSubMatrix(is1, is1)
            #self._Ap.setOption(PETSc.Mat.Option.SPD, True)
            self._ksp_Ap.setOperators(self._Ap)
        elif not hasattr(self, "_Ap"):
            self._raise_attribute_error("Ap")
        # Update Kp
        if Kp is not None:
            self._Kp = dolfin.as_backend_type(Kp).mat().getSubMatrix(is1, is1)
        elif not hasattr(self, "_Kp"):
            self._raise_attribute_error("Kp")
        # Update Mp
        if Mp is not None:
            self._Mp = dolfin.as_backend_type(Mp).mat().getSubMatrix(is1, is1)
            #self._Mp.setOption(PETSc.Mat.Option.SPD, True)
            self._ksp_Mp.setOperators(self._Mp)
        elif not hasattr(self, "_Mp"):
            self._raise_attribute_error("Mp")
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
