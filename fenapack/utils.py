from dolfin import MPI, has_lu_solver_method
from petsc4py import PETSc

import functools


def get_default_factor_solver_type(comm):
    """Return first available factor solver type name.
    This is implemened using DOLFIN now."""

    methods_parallel = ("mumps", "superlu_dist", "pastix")
    methods_sequential = ("mumps", "umfpack", "superlu",
                          "superlu_dist", "pastix")

    if isinstance(comm, PETSc.Comm):
        comm = comm.tompi4py()

    if MPI.size(comm) > 1:
        methods = methods_parallel
    else:
        methods = methods_sequential

    for method in methods:
        if has_lu_solver_method(method):
            return method

    raise RuntimeError("Did not find any suitable direct sparse solver in PETSc")


def pc_set_factor_solver_type(pc, type):
    if PETSc.Sys.getVersion()[0:2] <= (3, 8) and PETSc.Sys.getVersionInfo()['release']:
        pc.setFactorSolverPackage(type)
    else:
        pc.setFactorSolverType(type)


def allow_only_one_call(func):
    """Decorator allowing provided instancemethod to
    be called only once. Additional calls raise error."""

    sentinel_variable_name = "___called_{}_".format(func.__name__)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if already called and raise
        if getattr(self, sentinel_variable_name, False):
            try:
                name = ".".join((__package__, self.__module__,
                                 self.__class__.__name__, func.__name__))
            except Exception:
                name = func.__name__
            raise RuntimeError("Multiple calls to {} not allowed".format(name))

        # Mark the function as called
        setattr(self, sentinel_variable_name, True)

        # Call the function
        return func(self, *args, **kwargs)

    return wrapper
