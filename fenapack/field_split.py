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

__all__ = ['FieldSplitSolver']

# -----------------------------------------------------------------------------
# Function converting dofs (dolfin) to IS (PETSc)
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
    dolfin.compile_extension_module(dofmap_dofs_is_cpp_code).dofmap_dofs_is
del dofmap_dofs_is_cpp_code
# -----------------------------------------------------------------------------

class FieldSplitSolver(dolfin.PETScKrylovSolver):
    """This class derives from 'dolfin.PETScKrylovSolver' and implements
    field split preconditioner for saddle point problems like incompressible
    [Navier-]Stokes flow."""

    def __init__(self, space, method):
        """Create field split solver on a given space for a particular method.

        *Arguments*
            space (:py:class:`FunctionSpace <dolfin.functions.functionspace>`)
                Mixed function space determining the field split.
            method
                Type of a PETSc KSP object, see help(PETSc.KSP.Type).
        """
        # Create KSP
        ksp = self._ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType(method)
        # Init parent class
        dolfin.PETScKrylovSolver.__init__(self, ksp)
        # Set up FIELDSPLIT preconditioning
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        self._is0 = dofmap_dofs_is(space.sub(0).dofmap())
        self._is1 = dofmap_dofs_is(space.sub(1).dofmap())
        pc.setFieldSplitIS(["u", self._is0], ["p", self._is1])
        # Initiate option databases for subsolvers
        self._OptDB_00 = PETSc.Options("fieldsplit_u_")
        self._OptDB_11 = PETSc.Options("fieldsplit_p_")
        # Set default parameter values
        self.parameters = self.default_parameters()

    def default_parameters(self):
        """Extend default parameter set of parent class."""
        # Get default parameters for parent class
        prm = dolfin.PETScKrylovSolver().default_parameters()
        # Add new parameters
        prm["preconditioner"].add("side", "right",
                                  ["left", "right", "symmetric"])
        prm_fs = dolfin.Parameters("fieldsplit")
        prm_fs.add("type", "schur",
                   ["additive", "multiplicative", "symmetric_multiplicative",
                    "special", "schur"])
        prm_fs.add(dolfin.Parameters("schur"))
        prm_fs["schur"].add("fact_type", "upper",
                            ["diag", "lower", "upper", "full"])
        prm_fs["schur"].add("precondition", "user",
                            ["self", "selfp", "a11", "user", "full"])
        # Add new parameters to 'petsc_krylov_solver' parameters
        prm["preconditioner"].add(prm_fs)
        return prm

    def get_subopts(self):
        """Return option databases enabling to set up subsolvers."""
        return self._OptDB_00, self._OptDB_11

    # Discard PETScKrylovSolver::set_operator() method
    def set_operator(self, A):
        raise NotImplementedError

    # Overload PETScKrylovSolver::set_operators() method
    def set_operators(self, *args, **kwargs):
        A = args[0] # system operator
        P = args[1] # preconditioning operator
        dolfin.PETScKrylovSolver.set_operators(self, A, P)
        #assert self._ksp.getOperators() == \
        #  (dolfin.as_backend_type(A).mat(), dolfin.as_backend_type(P).mat())
        self._custom_ksp_setup(*args[2:], **kwargs)

    def _custom_ksp_setup(self, *args, **kwargs):
        # Update global option database
        self._set_from_parameters()
        # Set up KSP
        self._ksp.setFromOptions()
        self._ksp.setUp() # NOTE: this includes operations within 'PCSetUp'
        # Get subKSP and subPC objects
        ksp0, ksp1 = self._ksp.getPC().getFieldSplitSubKSP()
        pc0, pc1 = ksp0.getPC(), ksp1.getPC()
        # Check if python context has been set up to define approximation of
        # 11-block inverse. If so, use *args, **kwargs to set up this context.
        if self._OptDB_11.hasName("pc_python_type"):
            ctx = pc1.getPythonContext()
            ctx.set_operators(self._is0, self._is1, *args, **kwargs)
        # # Set up each subPC explicitly before calling 'self.solve'. In such
        # # a case, the time needed for setup is not included in timings under
        # # "PETSc Krylov Solver".
        # timer = dolfin.Timer("FENaPack: set up subPC objects")
        # timer.start()
        # pc0.setUp()
        # pc1.setUp()
        # timer.stop()

    def _set_from_parameters(self):
        """Set up extra parameters added to parent class."""
        # Get access to global option database
        OptDB = PETSc.Options()
        # Add extra solver parameters to the global option database
        prm = self.parameters["preconditioner"]
        OptDB["ksp_pc_side"] = \
          prm["side"]
        OptDB["pc_fieldsplit_type"] = \
          prm["fieldsplit"]["type"]
        OptDB["pc_fieldsplit_schur_fact_type"] = \
          prm["fieldsplit"]["schur"]["fact_type"]
        OptDB["pc_fieldsplit_schur_precondition"] = \
          prm["fieldsplit"]["schur"]["precondition"]

if __name__ == "__main__":
    # TODO: Write proper unit test.
    dolfin.info("Tests of %s." % __file__)
    mesh = dolfin.UnitSquareMesh(2, 2)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    Q = dolfin.FunctionSpace(mesh, "CG", 1)
    W = V * Q
    solver = FieldSplitSolver(W, "gmres")
    dolfin.info(solver.parameters, True)
