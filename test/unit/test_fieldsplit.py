import pytest
from dolfin import *
from petsc4py import PETSc

import os

from fenapack import PCDKrylovSolver
from fenapack._field_split_utils import dofmap_dofs_is

from bench.test_pcd_scaling import create_forms
from bench.test_pcd_scaling import create_pcd_problem
from bench.test_pcd_scaling import get_random_string


def create_dummy_solver_and_problem():
    mesh = UnitSquareMesh(3, 3)
    class Gamma1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, P2*P1)
    ff = MeshFunction('size_t', W.mesh(), W.mesh().topology().dim()-1)
    ff.set_all(0)
    Gamma1().mark(ff, 1)
    w = Function(W)
    F, bcs, J, J_pc = create_forms(w, ff, 1.0, 1.0, "newton", "direct")
    problem = create_pcd_problem(F, bcs, J, J_pc, w, 1.0, ff, "BRM1")
    solver = PCDKrylovSolver(comm=mesh.mpi_comm())
    solver.set_options_prefix("s"+get_random_string()+"_")
    A = PETScMatrix(solver.mpi_comm())
    problem.J(A, w.vector())
    solver.set_operators(A, A)
    return solver, problem


@pytest.fixture(scope="module")
def petsc_use_debug():
    arch = os.environ.get("PETSC_ARCH", "")
    dir = os.environ.get("PETSC_DIR", "")
    var = arch + ":" + dir
    var = var.lower()
    return "dbg" in var or "debug" in var


@pytest.mark.skipif(not petsc_use_debug(), reason="segfaults on PETSc without debug")
@pytest.mark.xfail(reason="PETSc issue #160", strict=True, raises=PETSc.Error)
def test_petsc_issue_160_1():
    """Test that whether PETSc issue #160 is resolved and we
    should remove any workarounds"""
    solver, _ = create_dummy_solver_and_problem()
    solver.ksp().pc.getFieldSplitSubKSP()


@pytest.mark.skipif(not petsc_use_debug(), reason="segfaults on PETSc without debug")
@pytest.mark.xfail(reason="PETSc issue #160", strict=True, raises=(PETSc.Error, AssertionError))
def test_petsc_issue_160_2():
    """Test that whether PETSc issue #160 is resolved and we
    should remove any workarounds"""
    solver, _ = create_dummy_solver_and_problem()
    solver.set_options_prefix("foo_")
    assert solver.ksp().pc.getFieldSplitSubKSP()[0].getOptionsPrefix() == "foo_fieldsplit_u_"
    assert solver.ksp().pc.getFieldSplitSubKSP()[1].getOptionsPrefix() == "foo_fieldsplit_p_"


@pytest.mark.xfail(reason="PETSc issue #160", strict=True, raises=AssertionError)
def test_petsc_issue_160_3():
    """Test that whether PETSc issue #160 is resolved and we
    should remove any workarounds"""
    solver, problem = create_dummy_solver_and_problem()
    prefix = solver.get_options_prefix()
    PETSc.Options().setValue(prefix+"fieldsplit_u_ksp_type", "pipecr")

    # Setup fieldsplit PC (avoid init_pcd which calls setFromOptions)
    is0 = dofmap_dofs_is(problem.pcd_assembler.function_space().sub(0).dofmap())
    is1 = dofmap_dofs_is(problem.pcd_assembler.function_space().sub(1).dofmap())
    solver.ksp().pc.setFieldSplitIS(['u', is0], ['p', is1])
    solver.ksp().pc.setUp()

    # This should give false becase nobody called setFromOptions
    assert solver.ksp().pc.getFieldSplitSubKSP()[0].type != "pipecr"


def test_set_options_prefix():
    """Test that out workaround of PETSc issue #160 works"""
    solver, problem = create_dummy_solver_and_problem()

    # Check that setting prefix early works
    solver.set_options_prefix("foo_")
    solver.init_pcd(problem.pcd_assembler)
    assert solver.ksp().pc.getFieldSplitSubKSP()[0].getOptionsPrefix() == "foo_fieldsplit_u_"
    assert solver.ksp().pc.getFieldSplitSubKSP()[1].getOptionsPrefix() == "foo_fieldsplit_p_"

    # Check that setting prefix late raises
    with pytest.raises(RuntimeError):
        solver.set_options_prefix("bar_")
