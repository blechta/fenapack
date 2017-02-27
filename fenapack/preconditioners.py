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
        self.ksp_Ap = self.create_default_ksp(pc.comm)
        self.ksp_Mp = self.create_default_ksp(pc.comm)

        options_prefix = pc.getOptionsPrefix()
        self.ksp_Ap.setOptionsPrefix(options_prefix + "PCD_Ap_")
        self.ksp_Mp.setOptionsPrefix(options_prefix + "PCD_Mp_")


    def setFromOptions(self, pc):
        self.ksp_Ap.setFromOptions()
        self.ksp_Mp.setFromOptions()


    @staticmethod
    def create_default_ksp(comm):
        """Return Cholesky factorization KSP"""
        ksp = PETSc.KSP().create(comm)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.pc.setType(PETSc.PC.Type.CHOLESKY)
        # FIXME: Have utility function looking for mumps, superlu, etc.
        ksp.pc.setFactorSolverPackage("mumps")
        return ksp


    def get_work_vecs(self, v, num):
        """Return ``num`` of work vecs initially duplicated from v"""
        try:
            vecs = self._work_vecs
            assert len(vecs) == num
        except AttributeError:
            self._work_vecs = vecs = tuple(v.duplicate() for i in range(num))
        except AssertionError:
            raise ValueError("Changing number of work vecs not allowed")
        return vecs


    def init_pcd(self, pcd_problem):
        """Initialize by PCDProblem instance"""
        if hasattr(self, "problem"):
            raise RuntimeError("Reinitialization of PCDPC not allowed")
        self.problem = pcd_problem



class PCDPC_BRM1(BasePCDPC):
    """This class implements a modification of PCD variant similar to one by
    [2]_.

    .. [2] Olshanskii M. A., Vassilevski Y. V., *Pressure Schur complement
           preconditioners for the discrete Oseen problem*.
           SIAM J. Sci. Comput., 29(6), 2686-2704. 2007.
    """

    @dolfin.timed("FENaPack: PCDPC_BRM1 apply")
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
        # Fetch work vector
        z, = self.get_work_vecs(x, 1)

        # Apply PCD
        x.copy(result=z)        # z = x
        self.bcs_applier(z)     # apply bcs to z
        self.ksp_Ap.solve(z, y) # y = A_p^{-1} z
        self.mat_Kp.mult(y, z)  # z = K_p y
        z.axpy(1.0, x)          # z = z + x
        self.ksp_Mp.solve(z, y) # y = M_p^{-1} z
        # FIXME: How is with the sign bussines?
        y.scale(-1.0)           # y = -y


    def setUp(self, pc):
        # FIXME: Maybe move Mp and Ap setup to init and remove logic in backend.
        #        This will make it obvious that this is done once.
        self.problem.setup_ksp_Mp(self.ksp_Mp)
        self.problem.setup_ksp_Ap(self.ksp_Ap)
        self.mat_Kp = self.problem.setup_mat_Kp(
                mat=getattr(self, "mat_Kp", None))
        self.bcs_applier = self.problem.apply_pcd_bcs



class PCDPC_BRM2(BasePCDPC):
    """This class implements a modification steady variant of PCD
    described in [1]_.

    .. [1] Elman H. C., Silvester D. J., Wathen A. J., *Finite Elements and Fast
           Iterative Solvers: With Application in Incompressible Fluid Dynamics*.
           Oxford University Press 2005. 2nd edition 2014.
    """

    @dolfin.timed("FENaPack: PCDPC_BRM2 apply")
    def apply(self, pc, x, y):
        # FIXME: Fix the docstring
        """This method implements the action of the inverse of the approximate
        Schur complement :math:`\hat{S} = -M_p F_p^{-1} A_p`, that is

        .. math::

            y = \hat{S}^{-1} x = - (I + A_p^{-1} K_p) M_p^{-1} x.
        """
        # Fetch work vector
        z0, z1 = self.get_work_vecs(x, 2)

        # Apply PCD
        self.ksp_Mp.solve(x, y)   # y = M_p^{-1} x
        y.copy(result=z0)         # z0 = y
        self.mat_Kp.mult(z0, z1)  # z1 = K_p z0
        self.bcs_applier(z1)      # apply bcs to z1
        self.ksp_Ap.solve(z1, z0) # z0 = A_p^{-1} z1
        y.axpy(1.0, z0)           # y = y + z0
        # FIXME: How is with the sign bussines?
        y.scale(-1.0)             # y = -y


    def setUp(self, pc):
        # FIXME: Maybe move Mp and Ap setup to init and remove logic in backend.
        #        This will make it obvious that this is done once.
        self.problem.setup_ksp_Mp(self.ksp_Mp)
        self.problem.setup_ksp_Ap(self.ksp_Ap)
        self.mat_Kp = self.problem.setup_mat_Kp(
                mat=getattr(self, "mat_Kp", None))
        self.bcs_applier = self.problem.apply_pcd_bcs
