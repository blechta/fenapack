# Copyright (C) 2014-2017 Jan Blechta and Martin Rehor
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


class BasePCDPC(object):
    """Base python context for pressure convection diffusion (PCD)
    preconditioners."""
    def create(self, pc):
        self._ksp_Ap = self._prepare_default_ksp(pc.comm)
        self._ksp_Mp = self._prepare_default_ksp(pc.comm)


    def setFromOptions(self, pc):
        options_prefix = pc.getOptionsPrefix()

        # Update Ap
        self._ksp_Ap.setOptionsPrefix(options_prefix+"PCD_Ap_")
        self._ksp_Ap.setFromOptions()

        # Update Mp
        self._ksp_Mp.setOptionsPrefix(options_prefix+"PCD_Mp_")
        self._ksp_Mp.setFromOptions()


    def _raise_attribute_error(self, attr):
        raise AttributeError(
            "Attribute '_%s' must be set using '%s.set_operators()'"
            % (attr, type(self).__name__))


    @staticmethod
    def _prepare_default_ksp(comm):
        ksp = PETSc.KSP().create(comm)
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



class PCDPC_SEW(BasePCDPC):
    """This class implements steady variant of PCD described in [1]_.

    .. [1] Elman H. C., Silvester D. J., Wathen A. J., *Finite Elements and Fast
           Iterative Solvers: With Application in Incompressible Fluid Dynamics*.
           Oxford University Press 2005. 2nd edition 2014.
    """
    def apply(self, pc, x, y):
        """This method implements the action of the inverse of the approximate
        Schur complement :math:`\hat{S} = -M_p F_p^{-1} A_p`, that is

        .. math::

            y = \hat{S}^{-1} x = -A_p^{-1} F_p M_p^{-1} x.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_SEW apply")

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
            is0, is1 (:py:class:`petsc4py.PETSc.IS`)
                The index sets defining blocks in the field-splitted matrix.
            A (:py:class:`GenericMatrix`)
                Dummy system matrix. Not used by this implementation.
            P (:py:class:`GenericMatrix`)
                Dummy preconditioning matrix. Not used by this implementation.

        *Keyword arguments*
            Mp (:py:class:`GenericMatrix`)
                The matrix containing pressure mass matrix as its 11-block.
            Ap (:py:class:`GenericMatrix`)
                The matrix containing pressure laplacian as its 11-block.
            Fp (:py:class:`GenericMatrix`)
                The matrix containing pressure convection-diffusion as its
                11-block.
            bcs (:py:class:`DirichletBC`)
                List of boundary conditions that will be "applied" on
                :math:`A_p` and :math:`F_p`.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_SEW set_operators")

        # Prepare bcs for adjusting field split matrix
        if bcs is not None:
            if not hasattr(bcs, "__iter__"):
                bcs = [bcs]
            self._bcs = [SubfieldBC(bc, is1) for bc in bcs]
        elif not hasattr(self, "_bcs"):
            self._raise_attribute_error("bcs")

        # Update Ap
        if Ap is not None:
            self._Ap = Ap.mat().getSubMatrix(is1, is1, submat=getattr(self, '_Ap', None))
            # Apply boundary conditions along outflow boundary
            for bc in self._bcs:
                bc.apply_fdm(self._Ap)
            self._ksp_Ap.setOperators(self._Ap)
        elif not hasattr(self, "_Ap"):
            self._raise_attribute_error("Ap")

        # Update Fp
        if Fp is not None:
            self._Fp = Fp.mat().getSubMatrix(is1, is1, submat=getattr(self, '_Fp', None))
            # Apply boundary conditions along outflow boundary
            for bc in self._bcs:
                bc.apply_fdm(self._Fp)
        elif not hasattr(self, "_Fp"):
            self._raise_attribute_error("Fp")

        # Update Mp
        if Mp:
            self._Mp = Mp.mat().getSubMatrix(is1, is1, submat=getattr(self, '_Mp', None))
            self._Mp.setOption(PETSc.Mat.Option.SPD, True)
            self._ksp_Mp.setOperators(self._Mp)
        elif not hasattr(self, "_Mp"):
            self._raise_attribute_error("Mp")

        timer.stop()



class UnsteadyPCDPC_SEW(PCDPC_SEW):
    r"""This class implements variant of PCD described in [1]_ appropriate for
    unsteady problems. It derives from :py:class:`PCDPC_SEW` but pressure Laplacian
    :math:`A_p` is approximated by :math:`B (\operatorname{diag} M_u)^{-1}
    B^\top`, where :math:`M_u` is the velocity mass matrix, :math:`B`
    corresponds to 10-block of the system matrix (:math:`\operatorname{div}`
    operator) and :math:`B^\top` is its transpose, i.e. 01-block of the system
    matrix (:math:`\operatorname{grad}` operator).

    Note that :math:`B^\top` contains zero rows corresponding to dofs on the
    Dirichlet boundary since bcs have been applied on :math:`A`. Moreover,
    :math:`B (\operatorname{diag} M_u)^{-1} B^\top` is nonsingular for
    inflow-outflow problems, thus we do not need to prescribe any artificial
    boundary conditions for pressure.
    """
    def set_operators(self, is0, is1, A, P,
                      Mp=None, Mu=None, Fp=None, bcs=None):
        """Set operators for the approximate Schur complement matrix

        *Arguments*
            is0, is1 (:py:class:`petsc4py.PETSc.IS`)
                The index sets defining blocks in the field splitted matrix.
            A (:py:class:`GenericMatrix`)
                The system matrix.
            P (:py:class:`GenericMatrix`)
                Dummy preconditioning matrix. Not used by this implementation.

        *Keyword arguments*
            Mp (:py:class:`GenericMatrix`)
                The matrix containing pressure mass matrix as its 11-block.
            Mu (:py:class:`GenericMatrix`)
                The matrix containing velocity mass matrix as its 00-block.
            Fp (:py:class:`GenericMatrix`)
                The matrix containing pressure convection-diffusion as its
                11-block.
            bcs (:py:class:`DirichletBC`)
                List of boundary conditions that will be "applied" on
                :math:`F_p`.
        """
        timer = dolfin.Timer("FENaPack: UnsteadyPCDPC_SEW set_operators")

        # Assemble Ap using A and Mu
        if Mu is not None:
            # Get velocity mass matrix as PETSc Mat object
            Mu = Mu.mat().getSubMatrix(is0, is0)

            # Get diagonal of the velocity mass matrix
            diagMu = Mu.getDiagonal()

            # Make inverse of diag(Mu)
            diagMu.reciprocal() # diag(Mu)^{-1}

            # Make square root of the diagonal and use it for scaling
            diagMu.sqrtabs() # \sqrt{diag(Mu)^{-1}}

            # Extract 01-block, i.e. "grad", from the matrix operator A
            Bt = A.mat().getSubMatrix(is0, is1)
            Bt.diagonalScale(L=diagMu) # scale rows of Bt

            # Get Ap
            self._Ap = Bt.transposeMatMult(Bt) # Ap = Bt^T*Bt

            # Set up special options
            self._Ap.setOption(PETSc.Mat.Option.SPD, True)

            # Store matrix in the corresponding ksp
            self._ksp_Ap.setOperators(self._Ap)

        elif not hasattr(self, "_Ap"):
            self._raise_attribute_error("Ap")

        # Update remaining operators in the same way as in the parent class
        PCDPC_SEW.set_operators(self, is0, is1, A, P, Mp=Mp, Fp=Fp, bcs=bcs)

        timer.stop()



class PCDPC_BRM(BasePCDPC):
    """This class implements a modification of PCD variant similar to one by
    [2]_.

    .. [2] Olshanskii M. A., Vassilevski Y. V., *Pressure Schur complement
           preconditioners for the discrete Oseen problem*.
           SIAM J. Sci. Comput., 29(6), 2686-2704. 2007.
    """
    def __init__(self):
        self._Ap = None


    def apply(self, pc, x, y):
        r"""This method implements the action of the inverse of the approximate
        Schur complement :math:`-\hat{S}^{-1}`, that is

        .. math::

            y = -M_p^{-1} (I + K_p A_p^{-1}) x

        where :math:`K_p` is used to denote pressure convection matrix plus
        possibly pressure mass matrix coming from discrete time derivative.
        Note that Laplace solve with :math:`A_p^{-1} x` is performed with usual
        non-symmetric application of subfield BC on matrix :math:`A_p` and RHS
        :math:`x` (but only in that term). It is crucial that identity term
        :math:`I x` is not absorbed into the second, compact term to get

        .. math::

            y = -M_p^{-1} (A_p + K_p) A_p^{-1} x.

        This is crucial to keep a stability with respect to the leading Stokes
        term.

        Good strategy is to use :math:`M_p` and :math:`K_p` both scaled by
        :math:`\nu^{-1}`.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_BRM apply")

        # Fetch cached duplicate of x
        z = self._z(x)
        x.copy(result=z)

        # Apply subfield boundary conditions to rhs
        for bc in self._subfield_bcs:
            bc.apply(z)

        # Apply PCD
        self._ksp_Ap.solve(z, y) # y = A_p^{-1} z
        self._Kp.mult(-y, z)     # z = -K_p y
        z.axpy(-1.0, x)          # z = z - x
        self._ksp_Mp.solve(z, y) # y = M_p^{-1} z

        timer.stop()


    def set_operators(self, is0, is1, A, P,
                      Mp=None, Ap=None, Kp=None, bcs=None):
        """Set operators for the approximate Schur complement matrix

        *Arguments*
            is0, is1 (:py:class:`petsc4py.PETSc.IS`)
                The index sets defining blocks in the field splitted matrix.
            A (:py:class:`GenericMatrix`)
                Dummy system matrix. Not used by this implementation.
            P (:py:class:`GenericMatrix`)
                Dummy preconditioning matrix. Not used by this implementation.

        *Keyword arguments*
            Mp (:py:class:`GenericMatrix`)
                The matrix containing pressure mass matrix as its 11-block.
            Ap (:py:class:`GenericMatrix`)
                The matrix containing pressure laplacian as its 11-block.
            Kp (:py:class:`GenericMatrix`)
                The matrix containing pressure convection plus possibly mass
                matrix comming from the time derivative as its 11-block.
            bcs (:py:class:`DirichletBC`)
                List of boundary conditions that will be applied during
                :math:`A_p^{-1} x` solve.
        """
        timer = dolfin.Timer("FENaPack: PCDPC_BRM set_operators")

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
            # NOTE: BC might already have been applied but this is not
            #       harmful. If symmetric approach (using SystemAssembler)
            #       was used then only homogeneous BC makes sense
            for bc in self._bcs:
                bc.apply(Ap)
            self._Ap = Ap.mat().getSubMatrix(is1, is1, submat=getattr(self, '_Ap', None))
            self._ksp_Ap.setOperators(self._Ap)
        elif not hasattr(self, "_Ap"):
            self._raise_attribute_error("Ap")

        # Update Kp
        if Kp is not None:
            # NOTE: getSubMatrix is deep, createSubMatrix is shallow
            # FIXME: Need a proper interface, this is really hacky!
            #self._Kp = Kp.mat().getSubMatrix(is1, is1, submat=getattr(self, '_Kp', None))
            if not hasattr(self, "_Kp"):
                # Need to createSubMatrix only once
                self._Kp = PETSc.Mat().createSubMatrix(Kp.mat(), is1, is1)
        elif not hasattr(self, "_Kp"):
            self._raise_attribute_error("Kp")

        # Update Mp
        if Mp is not None:
            self._Mp = Mp.mat().getSubMatrix(is1, is1, submat=getattr(self, '_Mp', None))
            self._Mp.setOption(PETSc.Mat.Option.SPD, True)
            self._ksp_Mp.setOperators(self._Mp)
        elif not hasattr(self, "_Mp"):
            self._raise_attribute_error("Mp")

        timer.stop()
