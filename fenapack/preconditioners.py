# Copyright (C) 2014-2016 Jan Blechta and Martin Rehor
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

        return ksp


    def _prepare_Mp_fact(self):
        # Prepare Mp factorization
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)

        # Default settings
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)

        return ksp


    def _z(self, v):
        """Return cached duplicate of PETSc Vec v."""
        try:
            return self._z_vec
        except AttributeError:
            self._z_vec = v.duplicate()
            return self._z_vec



class PCDPC_ESW(BasePCDPC):
    """This class implements PCD preconditioning proposed by Elman, Silvester
    and Wathen (2014)."""
    def apply(self, pc, x, y):
        """This method implements the action of the inverse of the approximate
        Schur complement $\hat{S} = -M_p F_p^{-1} A_p$, that is

            $y = \hat{S}^{-1} x = -A_p^{-1} F_p M_p^{-1} x$.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_ESW apply")

        # Fetch cached duplicate of x
        z = self._z(x)

        # Apply PCD
        self._ksp_Mp.solve(x, y) # y = M_p^{-1} x
        self._Fp.mult(-y, z)     # z = -F_p y
        self._ksp_Ap.solve(z, y) # y = A_p^{-1} z

        timer.stop()


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

        # Apply subfield boundary conditions to rhs
        for bc in self._subfield_bcs:
            bc.apply(x)

        # Fetch cached duplicate of x
        z = self._z(x)

        # Apply PCD
        self._ksp_Ap.solve(x, y) # y = A_p^{-1} x
        self._Kp.mult(-y, z)
        z.axpy(-self._nu, x)    # z = -K_p y - nu*x
        self._ksp_Mp.solve(z, y) # y = M_p^{-1} z

        timer.stop()


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
