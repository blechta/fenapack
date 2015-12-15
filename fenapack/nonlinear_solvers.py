# Copyright (C) 2015 Jan Blechta and Martin Rehor
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

__all__ = ['NonlinearSolver', 'NonlinearDiscreteProblem']

class NonlinearSolver(dolfin.NewtonSolver):
    """This class derives from 'dolfin.NewtonSolver' and takes linear solver as
    the first input argument. Additional setup of the linear solver can be made
    via keyword argument named 'update_hook'."""

    def __init__(self, solver, update_hook=None):
        """Create nonlinear variational solver for given problem.

        *Arguments*
            solver (:py:class:`GenericLinearSolver`)
                A nonlinear variational problem.
            update_hook
                A function to update operators of linear solver during
                successive nonlinear steps. Provided function is called within
                'dolfin::NewtonSolver::solve' and takes the same arguments.
        """
        factory = dolfin.PETScFactory.instance()
        dolfin.NewtonSolver.__init__(self, solver, factory)
        # CONVENTION: Linear solver is provided with parameters that
        #             need no updates from outer nonlinear solver.
        self.parameters.remove("krylov_solver")
        self.parameters.remove("lu_solver")
        # This is a temporary hack due to the following bug in DOLFIN:
        #   dolfin::PETScKrylovSolver::parameters_type returns "default"
        #   instead of "krylov_solver".
        self.parameters.add(dolfin.Parameters("default"))
        # Store arguments
        self._hook = update_hook
        # Homebrewed implementation
        self._solver = solver
        self._matA = dolfin.Matrix()
        self._dx = dolfin.Vector()
        self._b = dolfin.Vector()
        self._residual = 0.0
        self._residual0 = 0.0

    def converged(self, r, problem, newton_iteration):
        # A Python rewrite of 'dolfin::NewtonSolver::converged'
        # with modified reporting message
        rtol = self.parameters["relative_tolerance"]
        atol = self.parameters["absolute_tolerance"]
        report = self.parameters["report"]
        self._residual = r.norm("l2")
        if newton_iteration == 0:
          self._residual0 = self._residual
        relative_residual = self._residual/self._residual0
        if report and dolfin.MPI.rank(dolfin.mpi_comm_world()) == 0:
          dolfin.info("Nonlinear iteration %d:"
                      " r (abs) = %.3e (tol = %.3e)"
                      " r (rel) = %.3e (tol = %.3e)"
                      % (newton_iteration,
                         self._residual, atol,
                         relative_residual, rtol))
        return relative_residual < rtol or self._residual < atol

    # Homebrewed implementation
    def solve(self, problem, x):
        # A Python rewrite of 'dolfin::NewtonSolver::solve'
        # with some slight modifications in linear solver setup
        convergence_criterion = self.parameters["convergence_criterion"]
        maxiter = self.parameters["maximum_iterations"]

        # MODIFICATION: Linear solver is provided to the constructor.
        # solver_type = self.parameters["linear_solver"]
        # pc_type = self.parameters["preconditioner"]
        # if not self._solver:
        #     self._solver = dolfin.LinearSolver(solver_type, pc_type)

        # MODIFICATION: Linear solver needs no updates from nonlinear solver.
        # self._solver.update_parameters(self.parameters[self._solver.parameter_type()])

        krylov_iterations = 0
        self._newton_iteration = 0

        problem.F(self._b, x)
        problem.form(self._matA, self._b, x)

        if convergence_criterion == "residual":
            newton_converged = self.converged(self._b, problem, 0)
        elif convergence_criterion == "incremental":
            newton_converged = False
        else:
            dolfin.dolfin_error("utils.py"
                                "check for convergence",
                                "The convergence criterion %s is unknown,"
                                " known criteria are 'residual' or 'incremental'"
                                % convergence_criterion)

        relaxation = self.parameters["relaxation_parameter"]

        while not newton_converged and self._newton_iteration < maxiter:
            # MODIFICATION
            if self._hook:
                dolfin.debug('Calling updating hook to set up linear solver at'
                             ' iteration %d' % self._newton_iteration)
                self._hook(problem, x)
            else:
                problem.J(self._matA, x)
                self._solver.set_operator(self._matA)

            if not self._dx.empty():
                self._dx.zero()
            krylov_iterations += self._solver.solve(self._dx, self._b)
            if abs(1.0 - relaxation) < dolfin.DOLFIN_EPS:
                x[:] -= self._dx
            else:
                x.axpy(-relaxation, self._dx)
            self._newton_iteration += 1
            problem.F(self._b, x)
            problem.form(self._matA, self._b, x)
            if convergence_criterion == "residual":
                newton_converged = self.converged(self._b, problem,
                                                  self._newton_iteration)
            elif convergence_criterion == "incremental":
                newton_converged = self.converged(self._dx, problem,
                                                  self._newton_iteration-1)
            else:
                dolfin.dolfin_error("utils.py"
                                    "check for convergence",
                                    "The convergence criterion %s is unknown,"
                                    " known criteria are 'residual' or 'incremental'"
                                    % convergence_criterion)

        if newton_converged:
            if dolfin.MPI.rank(dolfin.mpi_comm_world()) == 0:
                dolfin.info("Nonlinear solver finished in %d iterations"
                            " and %d linear solver iterations."
                            % (self._newton_iteration, krylov_iterations))
        else:
            error_on_nonconvergence = self.parameters["error_on_nonconvergence"]
            if error_on_nonconvergence:
                if self._newton_iteration == maxiter:
                    dolfin.dolfin_error("utils.py",
                                        "solve nonlinear system with NewtonSolver",
                                        "Newton solver did not converge because"
                                        " maximum number of iterations reached")
                else:
                    dolfin.dolfin_error("utils.py",
                                        "solve nonlinear system with NewtonSolver",
                                        "Newton solver did not converge")
            else:
                dolfin.warning("Newton solver did not converge.")

        return self._newton_iteration, newton_converged

class NonlinearDiscreteProblem(dolfin.NonlinearProblem):
    """Class for interfacing with nonlinear solver."""
    def __init__(self, F, J, bcs):
        dolfin.NonlinearProblem.__init__(self)
        self._F = F
        self._J = J
        self._bcs = bcs
    def F(self, b, x):
        dolfin.assemble(self._F, tensor=b)
        for bc in self._bcs:
            bc.apply(b, x)
    def J(self, A, x):
        dolfin.assemble(self._J, tensor=A)
        for bc in self._bcs:
            bc.apply(A)
