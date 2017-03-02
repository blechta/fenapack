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

from __future__ import print_function

from dolfin import Timer, PETScKrylovSolver
from petsc4py import PETSc

from fenapack._field_split_utils import dofmap_dofs_is
from fenapack.field_split_backend import PCDInterface
from fenapack.preconditioners import PCDPC_BRM1
from fenapack.utils import get_default_factor_solver_package
from fenapack.utils import allow_only_one_call


def _create_pcd_ksp(comm, is0, is1, ksp=None):
    if ksp is None:
        ksp = PETSc.KSP()

    ksp.create(comm)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.pc.setType(PETSc.PC.Type.FIELDSPLIT)
    ksp.pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    ksp.pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)
    ksp.pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER)
    ksp.pc.setFieldSplitIS(["u", is0], ["p", is1])

    return ksp


class PCDKSP(PETSc.KSP):
    """GMRES with right fieldsplit preconditioning using upper
    Schur factorization and PCD Schur complement approximation"""

    # NOTE: We are not able to overload PETSc.KSP methods in this
    # class. That would only be possible with KSPPYTHON type but
    # we do not need it.

    def __init__(self, function_space):
        """Initialize PCDKSP for given mixed DOLFIN function
        space"""

        super(PCDKSP, self).__init__()

        comm = function_space.mesh().mpi_comm()
        is0 = dofmap_dofs_is(function_space.sub(0).dofmap())
        is1 = dofmap_dofs_is(function_space.sub(1).dofmap())

        self._is0, self._is1 = is0, is1

        _create_pcd_ksp(comm, is0, is1, ksp=self)


    @allow_only_one_call
    def init_pcd(self, pcd_problem, pcd_pc_class=None):
        """Initialize from ``PCDProblem`` instance. Needs to be called
        after ``setOperators`` and ``setUp``. That's why two-phase
        initialization is needed: first ``__init__``, then ``init_pcd``.
        """

        # Setup fieldsplit preconditioner
        # NOTE: Hacky PCSetUp_FieldSplit calls setFromOptions on subKSPs
        pc_prefix = self.pc.getOptionsPrefix() or ""
        with Timer("FENaPack: PCDKSP PC {} setup".format(pc_prefix)):
            self.pc.setUp()

        # Extract fieldsplit subKSPs (only once self.pc is set up)
        ksp0, ksp1 = self.pc.getFieldSplitSubKSP()

        # Set some sensible defaults
        ksp0.setType(PETSc.KSP.Type.PREONLY)
        ksp0.pc.setType(PETSc.PC.Type.LU)
        ksp0.pc.setFactorSolverPackage(get_default_factor_solver_package(self.comm))
        ksp1.setType(PETSc.KSP.Type.PREONLY)
        ksp1.pc.setType(PETSc.PC.Type.PYTHON)

        # Setup 0,0-block pc so that we have accurate timing
        ksp0_pc_prefix = ksp0.pc.getOptionsPrefix()
        with Timer("FENaPack: {} setup".format(ksp0_pc_prefix)):
            ksp0.pc.setFromOptions()
            ksp0.pc.setUp()

        # FIXME: Why don't we let user do this? This would simplify things
        # Initialize PCD PC context
        pcd_pc_prefix = ksp1.pc.getOptionsPrefix()
        pcd_pc_opt = PETSc.Options(pcd_pc_prefix).getString("pc_python_type","")
        # Use PCDPC class given by option
        if pcd_pc_opt != "":
            ksp1.pc.setFromOptions()
            pcd_pc = ksp1.pc.getPythonContext()
        # Use PCDPC class specified as argument
        elif pcd_pc_class is not None:
            pcd_pc = pcd_pc_class()
            ksp1.pc.setPythonContext(pcd_pc)
            ksp1.pc.setFromOptions()
        # Use default PCDPC class
        else:
            pcd_pc = PCDPC_BRM1()
            ksp1.pc.setPythonContext(pcd_pc)
            ksp1.pc.setFromOptions()

        # Get backend implementation of PCDProblem
        # FIXME: Make me parameter
        deep_submats = False
        #deep_submats = True
        pcd_interface = PCDInterface(pcd_problem, self._is0, self._is1,
                                     deep_submats=deep_submats)
        del self._is0, self._is1

        # FIXME: Why don't we let user do this? This would simplify things
        # Provide assembling routines to PCD
        try:
            pcd_pc.init_pcd(pcd_interface)
        except Exception:
            print("Initialization of PCD PC from PCDProblem failed!")
            print("Maybe wrong PCD PC class or PCDProblem instance.")
            raise

        # Setup PCD PC so that we have accurate timing
        with Timer("FENaPack: {} setup".format(pcd_pc_prefix)):
            ksp1.pc.setUp()



class PCDKrylovSolver(PETScKrylovSolver):
    """GMRES with right fieldsplit preconditioning using upper
    Schur factorization and PCD Schur complement approximation"""

    def __init__(self, function_space):
        """Initialize using given mixed function space"""
        self._ksp = PCDKSP(function_space)
        super(PCDKrylovSolver, self).__init__(self._ksp)

    def ksp(self):
        return self._ksp

    def init_pcd(self, pcd_problem, pcd_pc_class=None):
        """Initialize from ``PCDProblem`` instance. Needs to be called
        after ``setOperators`` and ``setUp``. That's why two-phase
        initialization is needed: first ``__init__``, then ``init_pcd``.
        """
        self._ksp.init_pcd(pcd_problem, pcd_pc_class=pcd_pc_class)
