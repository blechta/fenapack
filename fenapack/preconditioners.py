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

from dolfin import timed
from petsc4py import PETSc

from fenapack.utils import get_default_factor_solver_type
from fenapack.utils import pc_set_factor_solver_type


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
        pc_set_factor_solver_type(ksp.pc, get_default_factor_solver_type(comm))
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


    def init_pcd(self, pcd_interface):
        """Initialize by PCDInterface instance"""
        if hasattr(self, "interface"):
            raise RuntimeError("Reinitialization of PCDPC not allowed")
        self.interface = pcd_interface


    def setUp(self, pc):
        # Prepare mass matrix and Laplacian solvers
        # NOTE: Called function ensures that assembly, submat extraction and
        #       ksp setup is done only once during preconditioner lifetime.
        self.interface.setup_ksp_Mp(self.ksp_Mp)
        self.interface.setup_ksp_Ap(self.ksp_Ap)

        # Prepare convection matrix
        Kp = self.interface.setup_mat_Kp(mat=getattr(self, "mat_Kp", None))
        if Kp is not None: # updated only if not constant
            self.mat_Kp = Kp
            self.mat_Kp.setOptionsPrefix(pc.getOptionsPrefix() + "PCD_Kp_")

        # Fetch bcs apply function
        self.bcs_applier = self.interface.apply_pcd_bcs



class PCDPC_BRM1(BasePCDPC):
    """This class implements a modification of PCD variant similar to one by
    [1]_.

    .. [1] Olshanskii M. A., Vassilevski Y. V., *Pressure Schur complement
           preconditioners for the discrete Oseen problem*.
           SIAM J. Sci. Comput., 29(6), 2686-2704. 2007.
    """

    @timed("FENaPack: PCDPC_BRM1 apply")
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



class PCDPC_BRM2(BasePCDPC):
    """This class implements a modification of steady variant of PCD
    described in [2]_.

    .. [2] Elman H. C., Silvester D. J., Wathen A. J., *Finite Elements and Fast
           Iterative Solvers: With Application in Incompressible Fluid Dynamics*.
           Oxford University Press 2005. 2nd edition 2014.
    """

    @timed("FENaPack: PCDPC_BRM2 apply")
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



class BasePCDRPC(BasePCDPC):
    """Base python context for pressure convection diffusion reaction (PCDR)
    preconditioners.
    """
    # TODO: Add demo/bench to demonstrate performance of PCDR
    def create(self, pc):
        super(BasePCDRPC, self).create(pc)
        self.ksp_Rp = self.create_default_ksp(pc.comm)

        options_prefix = pc.getOptionsPrefix()
        self.ksp_Rp.setOptionsPrefix(options_prefix + "PCD_Rp_")


    def setFromOptions(self, pc):
        super(BasePCDRPC, self).setFromOptions(pc)
        self.ksp_Rp.setFromOptions()


    def setUp(self, pc):
        # Prepare mass matrix and Laplacian solvers, convection matrix and bcs_applier
        super(BasePCDRPC, self).setUp(pc)

        # Prepare Laplacian solver based on velocity mass matrix
        # and discrete pressure gradient
        Mu = self.interface.setup_mat_Mu(mat=getattr(self, "mat_Mu", None))
        if Mu is not None: # updated only if not constant
            self.mat_Mu = Mu
            self.mat_Mu.setOptionsPrefix(pc.getOptionsPrefix() + "PCD_Mu_")

        Bt = self.interface.setup_mat_Bt(mat=getattr(self, "mat_Bt", None))
        if Bt is not None: # updated only if not constant
            self.mat_Bt = Bt
            self.mat_Bt.setOptionsPrefix(pc.getOptionsPrefix() + "PCD_Bt_")

        self.interface.setup_ksp_Rp(self.ksp_Rp, self.mat_Mu, self.mat_Bt)



class PCDRPC_BRM1(BasePCDRPC):
    """This class implements an extension of :py:class:`PCDPC_BRM1`.
    Here we add a reaction term into the preconditioner, so that is becomes
    PCDR (pressure-convection-diffusion-reaction) preconditioner. This
    particular variant is suitable for time-dependent problems, where the
    reaction term arises from the time derivative in the balance of momentum.
    """

    @timed("FENaPack: PCDRPC_BRM1 apply")
    def apply(self, pc, x, y):
        r"""This method implements the action of the inverse of the approximate
        Schur complement :math:`-\hat{S}^{-1}`, that is

        .. math::

            y = - R_p^{-1} x - M_p^{-1} (I + K_p A_p^{-1}) x

        where :math:`K_p` is used to denote pressure convection matrix, while
        :math:`R_p` originates in the discretized time derivative. Roughly
        speaking, :math:`R_p^{-1} x` corresponds to additional Laplace solve
        in which the minus Laplacian operator is approximated in a clever way
        based on the *velocity mass matrix* and *pressure gradient operator*.

        Based on experimental evidence, we can say that this particular
        variant performs better than :py:class:`PCDPC_BRM1` (with :math:`K_p`
        enriched by the pressure mass matrix from the discrete time
        derivative) applied to time-dependent problems.
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
        self.ksp_Rp.solve(x, z) # z = R_p^{-1} x
        y.axpy(1.0, z)          # y = y + z
        # FIXME: How is with the sign bussines?
        y.scale(-1.0)           # y = -y



class PCDRPC_BRM2(BasePCDRPC):
    """This class implements an extension of :py:class:`PCDPC_BRM2`.
    Here we add a reaction term into the preconditioner, so that is becomes
    PCDR (pressure-convection-diffusion-reaction) preconditioner. This
    particular variant is suitable for time-dependent problems, where the
    reaction term arises from the time derivative in the balance of momentum.
    """

    @timed("FENaPack: PCDRPC_BRM2 apply")
    def apply(self, pc, x, y):
        # FIXME: Fix the docstring
        """This method implements the action of the inverse of the approximate
        Schur complement :math:`-\hat{S}^{-1}`, that is

        .. math::

            y = - R_p^{-1} x - (I + A_p^{-1} K_p) M_p^{-1} x.

        where :math:`K_p` is used to denote pressure convection matrix, while
        :math:`R_p` originates in the discretized time derivative. Roughly
        speaking, :math:`R_p^{-1} x` corresponds to additional Laplace solve
        in which the minus Laplacian operator is approximated in a clever way
        based on the *velocity mass matrix* and *pressure gradient operator*.

        Based on experimental evidence, we can say that this particular
        variant performs better than :py:class:`PCDPC_BRM2` (with :math:`K_p`
        enriched by the pressure mass matrix from the discrete time
        derivative) applied to time-dependent problems.
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
        self.ksp_Rp.solve(x, z0)  # z0 = R_p^{-1} x
        y.axpy(1.0, z0)           # y = y + z0
        # FIXME: How is with the sign bussines?
        y.scale(-1.0)             # y = -y
