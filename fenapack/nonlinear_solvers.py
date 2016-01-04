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
    the first input argument plus one optional keyword argument 'debug_hook'
    which may specify a function executed on every convergence test during
    successive nonlinear iterations as defined by
      'dolfin.NewtonSolver.converged(GenericVector r, NonlinearProblem problem,
                                     size_t iteration)'.
    Provided function takes the same arguments."""

    def __init__(self, solver, debug_hook=None):
        """Create nonlinear variational solver for given problem.

        *Arguments*
            solver (:py:class:`GenericLinearSolver`)
                A nonlinear variational problem.
        """
        factory = dolfin.PETScFactory.instance()
        dolfin.NewtonSolver.__init__(self, solver, factory)
        # CONVENTION: Linear solver is provided with parameters that
        #             need no updates from outer nonlinear solver.
        self.parameters.remove("krylov_solver")
        self.parameters.remove("lu_solver")
        # This is temporary hack due to the following bug in DOLFIN:
        #   dolfin::PETScKrylovSolver::parameters_type returns "default"
        #   instead of "krylov_solver".
        self.parameters.add(dolfin.Parameters("default"))
        # Store arguments
        self._hook = debug_hook
        # Homebrewed implementation
        self._solver = solver
        self._matA = dolfin.Matrix()
        self._matP = dolfin.Matrix()
        self._matAp = dolfin.Matrix()
        self._matFp = dolfin.Matrix()
        self._matKp = dolfin.Matrix()
        self._matMp = dolfin.Matrix()
        self._dx = dolfin.Vector()
        self._b = dolfin.Vector()
        self._residual = 0.0
        self._residual0 = 0.0

    def converged(self, r, problem, newton_iteration):
        # A Python rewrite of 'dolfin::NewtonSolver::converged' with
        # possibility to call debugging hook.
        if self._hook:
            dolfin.debug('Calling debugging hook of NonlinearSolver::converged'
                         ' at iteration %d' % newton_iteration)
            self._hook(r, problem, newton_iteration)
        rtol = self.parameters["relative_tolerance"]
        atol = self.parameters["absolute_tolerance"]
        report = self.parameters["report"]
        self._residual = r.norm("l2")
        if newton_iteration == 0:
            self._residual0 = self._residual
        relative_residual = self._residual/self._residual0
        # Print modified report message
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
            problem.J(self._matA, x)
            # MODIFICATION: Custom setup of linear solver at each iteration step.
            #self._solver.set_operator(self._matA)
            setargs = [self._matA]
            try:
                problem.J_pc(self._matP, x)
                setargs.append(self._matP)
            except AttributeError: # J_pc == J
                setargs.append(self._matA)
            setkwargs = dict()
            for key in ["Fp", "Kp"]:
                try:
                    getattr(problem, key.lower())(getattr(self, "_mat"+key), x)
                    setkwargs[key] = getattr(self, "_mat"+key)
                except AttributeError:
                    pass
            if self._newton_iteration == 0:
                setkwargs["bcs"] = problem.bcs_pcd()
                if setkwargs.has_key("Kp"): # Hack for fenapack.PCDPC_BMR
                    setkwargs["nu"] = problem.nu()
                for key in ["Ap", "Mp"]:
                    try:
                        getattr(problem, key.lower())(getattr(self, "_mat"+key), x)
                        setkwargs[key] = getattr(self, "_mat"+key)
                    except AttributeError:
                        pass
            # Finally call 'set_operators'
            self._solver.set_operators(*setargs, **setkwargs)

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
    def __init__(self, F, bcs, J, J_pc=None, **kwargs):
        dolfin.NonlinearProblem.__init__(self)
        self._F = F
        self._bcs = bcs
        self._J = J
        if J_pc:
            self._J_pc = J_pc
        # Optional keyword arguments [ap, fp, kp, mp, bcs_pcd]
        for item in kwargs.items():
            setattr(self, "_"+item[0], item[1])
        # Check boundary conditions for PCD preconditioning
        try: # make sure that 'bcs_pcd' is a list
            if not isinstance(self._bcs_pcd, list):
                self._bcs_pcd = [self._bcs_pcd]
        except AttributeError: # 'bcs_pcd' has not been provided
            self._bcs_pcd = []
    def F(self, b, x):
        # b ... residual vector
        dolfin.assemble(self._F, tensor=b)
        for bc in self._bcs:
            bc.apply(b, x)
    def J(self, A, x):
        # A ... system matrix
        dolfin.assemble(self._J, tensor=A)
        for bc in self._bcs:
            bc.apply(A)
    def J_pc(self, P, x):
        # P ... preconditioning matrix
        dolfin.assemble(self._J_pc, tensor=P)
        for bc in self._bcs:
            bc.apply(P)
    def ap(self, Ap, x):
        # Ap ... pressure Laplacian matrix
        dolfin.assemble(self._ap, tensor=Ap)
    def fp(self, Fp, x):
        # Fp ... pressure convection-diffusion matrix
        dolfin.assemble(self._fp, tensor=Fp)
    def kp(self, Kp, x):
        # Kp ... pressure convection matrix
        dolfin.assemble(self._kp, tensor=Kp)
    def mp(self, Mp, x):
        # Mp ... pressure mass matrix
        dolfin.assemble(self._mp, tensor=Mp)
    def bcs_pcd(self):
        # Return boundary conditions for PCD preconditioning
        return self._bcs_pcd
    def nu(self):
        # Hack for fenapack.PCDPC_BMR (needs to know viscosity explicitly)
        return self._nu
