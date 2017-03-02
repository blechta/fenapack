# Copyright (C) 2015-2017 Jan Blechta and Martin Rehor
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

from dolfin import NewtonSolver, PETScFactory, NonlinearProblem
from dolfin import SystemAssembler, assemble

class PCDNewtonSolver(NewtonSolver):

    def __init__(self, solver):
        # Initialize DOLFIN Newton solver
        comm = solver.mpi_comm()
        factory = PETScFactory.instance()
        super(PCDNewtonSolver, self).__init__(comm, solver, factory)

        # Store Python reference for solver setup
        self._solver = solver


    def solve(self, problem, x):
        # Store Python reference for solver setup
        self._problem = problem

        # Solve the problem, drop the reference, and return
        r = super(PCDNewtonSolver, self).solve(problem, x)
        del self._problem
        return r


    def solver_setup(self, A, P, nonlinear_problem, iteration):
        # Only do the setup once
        # FIXME: Is this good?
        if iteration > 0 or getattr(self, "_initialized", False):
            return
        self._initialized = True

        # C++ references passed in do not have Python context
        linear_solver = self._solver
        nonlinear_problem = self._problem

        # Set operators and initialize PCD
        P = A if P.empty() else P
        linear_solver.set_operators(A, P)
        linear_solver.init_pcd(nonlinear_problem)



# FIXME: Separate Newton and PCD part of this class. Linear PCD problem
# has nothing to do with Newton part
class PCDProblem(NonlinearProblem):
    """Class for interfacing with not only :py:class:`NewtonSolver`."""
    # TODO: Add abstract base class with docstrings
    # TODO: What about interface
    #          pcd_problem = PCDProblem()
    #          pcd_problem.forms.F = F
    #          pcd_problem.forms.J = J
    #          ....
    def __init__(self, F, bcs, J, J_pc=None,
                 mp=None, mu=None, ap=None, fp=None, kp=None, bcs_pcd=[]):
        """Return subclass of :py:class:`dolfin.NonlinearProblem` suitable
        for :py:class:`NewtonSolver` based on
        :py:class:`fenapack.field_split.FieldSplitSolver` and PCD
        preconditioners from :py:class:`fenapack.field_split`.

        *Arguments*
            F (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Linear form representing the equation.
            bcs (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Boundary conditions applied to ``F``, ``J``, and ``J_pc``.
            J (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing system Jacobian.
            J_pc (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing Jacobian optionally passed to
                preconditioner instead of ``J``. In case of PCD, stabilized
                00-block can be passed to 00-KSP solver.
            mp, mu, ap, fp, kp (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear forms which (some of them) might be used by a
                particular PCD preconditioner. Typically they represent "mass
                matrix" on pressure, "mass matrix" on velocity, minus Laplacian
                operator on pressure, pressure convection-diffusion operator,
                and pressure convection operator respectively.

                ``mp``, ``mu``, and ``ap`` are assumed to be constant during
                subsequent non-linear iterations and are assembled only once.
                On the other hand, ``fp`` and ``kp`` are updated in every
                iteration.
            bcs_pcd (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Artificial boundary conditions used by PCD preconditioner.

        All the arguments should be given on the common mixed function space.
        """

        super(PCDProblem, self).__init__()

        # Assembler for Newton/Picard system
        # FIXME: Does it get broken for Oseen system?
        self.assembler = SystemAssembler(J, F, bcs)

        # Assembler for preconditioner
        if J_pc is not None:
            self.assembler_pc = SystemAssembler(J_pc, F, bcs)
        else:
            self.assembler_pc = None

        # Store forms/bcs for later
        self.forms = {
            "F": F,
            "ap": ap,
            "mp": mp,
            "mu": mu,
            "fp": fp,
            "kp": kp,
        }
        self._bcs_pcd = bcs_pcd


    def get_form(self, key):
        form = self.forms.get(key)
        if form is None:
            raise AttributeError("Form '%s' requested by PCD not available" % key)
        return form


    def F(self, b, x):
        self.assembler.assemble(b, x)


    def J(self, A, x):
        self.assembler.assemble(A)


    def J_pc(self, P, x):
        if self.assembler_pc is not None:
            self.assembler_pc.assemble(P)


    def ap(self, Ap):
        assembler = SystemAssembler(self.get_form("ap"), self.get_form("F"),
                                    self.pcd_bcs())
        assembler.assemble(Ap)


    def mp(self, Mp):
        assemble(self.get_form("mp"), tensor=Mp)


    def mu(self, Mu):
        assemble(self.get_form("mu"), tensor=Mu)


    def fp(self, Fp):
        assemble(self.get_form("fp"), tensor=Fp)


    def kp(self, Kp):
        assemble(self.get_form("kp"), tensor=Kp)


    # FIXME: Naming
    def pcd_bcs(self):
        try:
            assert self._bcs_pcd is not None
            return self._bcs_pcd
        except (AttributeError, AssertionError):
            raise AttributeError("PCD BCs requested by not available")
