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

"""This module provides subclasses of DOLFIN and petsc4py
Krylov solvers implementing PCD fieldsplit preconditioned
GMRES"""

from __future__ import print_function

from dolfin import Timer, PETScKrylovSolver
from petsc4py import PETSc
from mpi4py import MPI

from fenapack._field_split_utils import dofmap_dofs_is
from fenapack.field_split_backend import PCDInterface
from fenapack.preconditioners import PCDPC_BRM1
from fenapack.utils import get_default_factor_solver_type
from fenapack.utils import pc_set_factor_solver_type
from fenapack.utils import allow_only_one_call


class PCDKSP(PETSc.KSP):
    """GMRES with right fieldsplit preconditioning using upper
    Schur factorization and PCD Schur complement approximation.

    This is a subclass of ``petsc4py.KSP``."""

    # NOTE: We are not able to overload PETSc.KSP methods in this
    # class. That would only be possible with KSPPYTHON type but
    # we do not need it.

    def __init__(self, comm=None):
        """Initialize PCDKSP on given MPI comm"""

        super(PCDKSP, self).__init__()

        self.create(comm)
        self.setType(PETSc.KSP.Type.GMRES)
        self.setPCSide(PETSc.PC.Side.RIGHT)
        self.pc.setType(PETSc.PC.Type.FIELDSPLIT)
        self.pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        self.pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)
        self.pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER)


    @allow_only_one_call
    def init_pcd(self, pcd_assembler, pcd_pc_class=None):
        """Initialize from ``PCDAssembler`` instance. Needs to be called
        after ``setOperators`` and ``setUp``. That's why two-phase
        initialization is needed: first ``__init__``, then ``init_pcd``

        Note that this function automatically calls setFromOptions to
        all subKSP objects.
        """

        # Get subfield index sets
        V = pcd_assembler.function_space()
        is0 = dofmap_dofs_is(V.sub(0).dofmap())
        is1 = dofmap_dofs_is(V.sub(1).dofmap())

        comm = self.comm.tompi4py()
        assert comm.Compare(comm, V.mesh().mpi_comm()) in \
            [MPI.IDENT, MPI.CONGRUENT], "Non-matching MPI comm"

        # Set subfields index sets
        # NOTE: Doing only so late here so that user has a chance to
        # set options prefix, see PETSc issue #160
        self.pc.setFieldSplitIS(["u", is0], ["p", is1])

        # From now on forbid setting options prefix (at least from Python)
        self.setOptionsPrefix = self._forbid_setOptionsPrefix

        # Setup fieldsplit preconditioner
        pc_prefix = self.pc.getOptionsPrefix() or ""
        with Timer("FENaPack: PCDKSP PC {} setup".format(pc_prefix)):
            self.pc.setUp()

        # Extract fieldsplit subKSPs (only once self.pc is set up)
        ksp0, ksp1 = self.pc.getFieldSplitSubKSP()

        # Set some sensible defaults
        ksp0.setType(PETSc.KSP.Type.PREONLY)
        ksp0.pc.setType(PETSc.PC.Type.LU)
        pc_set_factor_solver_type(ksp0.pc, get_default_factor_solver_type(comm))
        ksp1.setType(PETSc.KSP.Type.PREONLY)
        ksp1.pc.setType(PETSc.PC.Type.PYTHON)

        # Setup 0,0-block pc so that we have accurate timing
        ksp0_prefix = ksp0.getOptionsPrefix()
        with Timer("FENaPack: {} setup".format(ksp0_prefix)):
            ksp0.setFromOptions()  # Override defaults above by user's options
            ksp0.pc.setUp()

        # Initialize PCD PC context
        pcd_pc_prefix = ksp1.pc.getOptionsPrefix()
        pcd_pc_opt = PETSc.Options(pcd_pc_prefix).getString("pc_python_type","")
        # Use PCDPC class given by option
        if pcd_pc_opt != "":
            ksp1.setFromOptions()  # Override defaults above by user's options
            pcd_pc = ksp1.pc.getPythonContext()
        # Use PCDPC class specified as argument
        elif pcd_pc_class is not None:
            pcd_pc = pcd_pc_class()
            ksp1.pc.setPythonContext(pcd_pc)
            ksp1.setFromOptions()  # Override defaults above by user's options
        # Use default PCDPC class
        else:
            pcd_pc = PCDPC_BRM1()
            ksp1.pc.setPythonContext(pcd_pc)
            ksp1.setFromOptions()  # Override defaults above by user's options

        # Get backend implementation of PCDAssembler
        # FIXME: Make me parameter
        #deep_submats = False
        deep_submats = True
        A = self.getOperators()[0]
        pcd_interface = PCDInterface(pcd_assembler, A, is0, is1,
                                     deep_submats=deep_submats)

        # Provide assembling routines to PCD
        try:
            pcd_pc.init_pcd(pcd_interface)
        except Exception:
            print("Initialization of PCD PC from PCDAssembler failed!")
            print("Maybe wrong PCD PC class or PCDAssembler instance.")
            raise

        # Setup PCD PC so that we have accurate timing
        with Timer("FENaPack: {} setup".format(pcd_pc_prefix)):
            ksp1.pc.setUp()


    def _forbid_setOptionsPrefix(self, prefix):
        raise RuntimeError("Options prefix cannot be set now. "
                           "Set it before init_pcd.")



class PCDKrylovSolver(PETScKrylovSolver):
    """GMRES with right fieldsplit preconditioning using upper
    Schur factorization and PCD Schur complement approximation.

    This is a subclass of ``dolfin.PETScKrylovSolver``."""

    def __init__(self, comm=None):
        """Initialize Krylov solver on given MPI comm"""
        self._ksp = PCDKSP(comm=comm)
        super(PCDKrylovSolver, self).__init__(self._ksp)


    def init_pcd(self, pcd_assembler, pcd_pc_class=None):
        """Initialize from ``PCDAssembler`` instance. Needs to be called
        after ``setOperators`` and ``setUp``. That's why two-phase
        initialization is needed: first ``__init__``, then ``init_pcd``

        Note that this function automatically calls setFromOptions to
        all subKSP objects.
        """
        self._ksp.init_pcd(pcd_assembler, pcd_pc_class=pcd_pc_class)


    def ksp(self):
        return self._ksp


    ksp.__doc__ = PETScKrylovSolver.ksp.__doc__


    def set_options_prefix(self, prefix):
        self._ksp.setOptionsPrefix(prefix)


    set_options_prefix.__doc__ = PETScKrylovSolver.set_options_prefix.__doc__
