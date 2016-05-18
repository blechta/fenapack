# Copyright (C) 2015-2016 Jan Blechta and Martin Rehor
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
import re

__all__ = ['NewtonSolver', 'PCDProblem']


class NewtonSolver(dolfin.NewtonSolver):
    """This class is a modification of :py:class:`dolfin.NewtonSolver`
    suitable to deal appropriately with
    :py:class:`fenapack.field_split.FieldSplitSolver` and PCD
    preconditioners from :py:class:`fenapack.preconditioners` module.
    In particular, it takes properly of linear solver and preconditioner
    setup in between succesive Newton iterations.
    """
    def __init__(self, solver, debug_hook=None):
        """Create Newton solver for solver for given problem. Optional
        debug hook executed on every iteration can be provided.

        *Arguments*
            solver (:py:class:`GenericLinearSolver`)
                A linear solver.
            debug_hook (function)
                A function of the signature::

                    bool dolfin.NewtonSolver.converged(GenericVector r, NonlinearProblem problem, int iteration)

                Provided ``r`` is a current residual vector, ``problem`` is
                argument supplied to ``solve`` and ``iteration`` is number of
                current iteration; :py:class:`bool` return value is ignored.
        """
        self._hook = debug_hook
        factory = dolfin.PETScFactory.instance()
        dolfin.NewtonSolver.__init__(self, solver.mpi_comm(), solver, factory)
        # Homebrewed implementation
        self._solver = solver
        self._matA = dolfin.Matrix()
        self._dx = dolfin.Vector()
        self._b = dolfin.Vector()
        self._residual = 0.0
        self._residual0 = 0.0


    def converged(self, r, problem, newton_iteration):
        # A Python rewrite of 'dolfin::NewtonSolver::converged' with
        # possibility to call debugging hook.
        if self._hook:
            dolfin.debug('Calling debugging hook of NewtonSolver::converged'
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
        """Solve abstract nonlinear problem :math:`F(x) = 0` for given
        :math:`F` and Jacobian :math:`\dfrac{\partial F}{\partial x}`.

        *Arguments*
            problem (:py:class:`dolfin.NonlinearProblem`)
                The nonlinear problem.
            x (:py:class:`dolfin.GenericVector`)
                The unknown vector.

        *Returns*
            (int, bool)
                Pair of number of Newton iterations, and whether
                iteration converged)
        """
        # A Python rewrite of 'dolfin::NewtonSolver::solve'
        # with modification in linear solver setup.
        convergence_criterion = self.parameters["convergence_criterion"]
        maxiter = self.parameters["maximum_iterations"]

        assert hasattr(self, "_solver")
        # solver_type = self.parameters["linear_solver"]
        # pc_type = self.parameters["preconditioner"]
        # if not self._solver:
        #     self._solver = dolfin.LinearSolver(solver_type, pc_type)

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
            # TODO: Fix this in DOLFIN. Setup of the linear solver should be
            # handled by some appropriate method of 'NonlinearProblem'.
            # By default, this method must call
            #   self._solver.set_operator(self._matA)
            # in order to retain original functionality of 'NewtonSolver'.
            # Users should be allowed to overload this method to customize
            # linear solver setup, e.g. to set preconditioner.
            problem.linear_solver_setup(
                self._solver, self._matA, self._newton_iteration)

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



class PCDProblem(dolfin.NonlinearProblem):
    """Class for interfacing with :py:class:`NewtonSolver`."""
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

        dolfin.NonlinearProblem.__init__(self)

        self._F = F
        self._bcs = bcs
        self._J = J
        self._J_pc = J_pc
        self._mp = mp
        self._mu = mu
        self._ap = ap
        self._fp = fp
        self._kp = kp
        self._bcs_pcd = bcs_pcd

        # Matrices used to assemble parts of the preconditioner
        # NOTE: Some of them may be unused.
        self._matP  = dolfin.Matrix()
        self._matMp = dolfin.Matrix()
        self._matMu = dolfin.Matrix()
        self._matAp = dolfin.Matrix()
        self._matFp = dolfin.Matrix()
        self._matKp = dolfin.Matrix()


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


    def J_pc(self, P):
        # P ... preconditioning matrix
        self._check_attr("J_pc")
        dolfin.assemble(self._J_pc, tensor=P)
        for bc in self._bcs:
            bc.apply(P)


    def mp(self, Mp):
        # Mp ... pressure mass matrix
        self._check_attr("mp")
        dolfin.assemble(self._mp, tensor=Mp)


    def mu(self, Mu):
        # Mu ... velocity mass matrix
        self._check_attr("mu")
        dolfin.assemble(self._mu, tensor=Mu)


    def ap(self, Ap):
        # Ap ... pressure Laplacian matrix
        self._check_attr("ap")
        dolfin.assemble(self._ap, tensor=Ap)


    def fp(self, Fp):
        # Fp ... pressure convection-diffusion matrix
        self._check_attr("fp")
        dolfin.assemble(self._fp, tensor=Fp)


    def kp(self, Kp):
        # Kp ... pressure convection matrix
        self._check_attr("kp")
        dolfin.assemble(self._kp, tensor=Kp)


    #Hook called by :py:class:`NewtonSolver` on every iteration.
    def linear_solver_setup(self, solver, A, it):
        # Assemble preconditioner Jacobian or use system Jacobian
        try:
            self.J_pc(self._matP)
            P = self._matP
        except AttributeError:
            P = A
        else:
            P = self._matP

        schur_approx = {}

        # Prepare matrices guaranteed to be constant during iterations once
        if it == 0:
            schur_approx["bcs"] = self._bcs_pcd
            for key in ["Ap", "Mp", "Mu"]:
                mat_object = getattr(self, "_mat"+key)
                try:
                    getattr(self, key.lower())(mat_object) # assemble
                except AttributeError:
                    pass
                else:
                    schur_approx[key] = mat_object

        # Assemble non-constant matrices everytime
        for key in ["Fp", "Kp"]:
            mat_object = getattr(self, "_mat"+key)
            try:
                getattr(self, key.lower())(mat_object) # assemble
            except AttributeError:
                pass
            else:
                schur_approx[key] = mat_object

        # Pass assembled operators and bc to linear solver
        solver.set_operators(A, P, **schur_approx)


    def _check_attr(self, attr):
        if getattr(self, "_"+attr) is None:
            raise AttributeError("Keyword argument '%s' of 'PCDProblem' object"
                                 " has not been set." % attr)
