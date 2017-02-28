import pytest

from dolfin import *

from fenapack.utils import get_default_factor_solver_package
from fenapack.utils import allow_only_one_call


def test_get_default_factor_solver_package():
    # Get available methods from DOLFIN
    methods = lu_solver_methods()
    methods.pop("default", None)

    # Test sequential method
    assert get_default_factor_solver_package(mpi_comm_self()) in methods

    # Test parallel method if in parallel
    comm_world = mpi_comm_world()
    if MPI.size(comm_world) > 1:
        methods.pop("umfpack", None)
        methods.pop("superlu", None)
    assert get_default_factor_solver_package(comm_world) in methods


def test_allow_only_one_call():
    class C():
        @allow_only_one_call
        def foo(self, *args, **kwargs):
            """Foo"""
            return args, kwargs

        @allow_only_one_call
        def bar(self, *args, **kwargs):
            """Bar"""
            return args, kwargs

        def baz(self, *args, **kwargs):
            """Baz"""
            return args, kwargs

    o = C()

    # Check docstrings are correct
    assert o.foo.__doc__ == "Foo"
    assert o.bar.__doc__ == "Bar"
    assert o.baz.__doc__ == "Baz"

    # Check first calls work as expected
    assert o.foo(1, 2, 3, four=5) == ((1, 2, 3), {'four': 5})
    assert o.bar(1, 2, 3, four=5) == ((1, 2, 3), {'four': 5})
    assert o.baz(1, 2, 3, four=5) == ((1, 2, 3), {'four': 5})

    # Check second calls raise
    with pytest.raises(RuntimeError):
        o.foo(1, 2, 3, four=5)
    with pytest.raises(RuntimeError):
        o.bar(1, 2, 3, four=5)
    assert o.baz(1, 2, 3, four=5) == ((1, 2, 3), {'four': 5})
