# -*- coding: utf-8 -*-

# Copyright (C) 2016-2017 Patrick Farrell, Jan Blechta
#
# This file comes originally from defcon.
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

# Let PETSc print memory usage at the end
import petsc4py
petsc4py.init(("python", "-malloc_info"))

from defcon import *
from dolfin import *
from matplotlib import pyplot

from fenapack import PCDKSP
from fenapack import PCDAssembler
from fenapack import StabilizationParameterSD

import sys
import argparse


class NavierStokesProblem(BifurcationProblem):
    def __init__(self):
        # FIXME: Hack for defcon gui
        argv = [] if __name__ == "prob" else sys.argv[1:]

        # Parse input arguments
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                         argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--pcd", type=str, dest="pcd", default="BRM2",
                            choices=["none", "BRM1", "BRM2"], help="PCD variant")
        parser.add_argument("--nls", type=str, dest="nls", default="newton",
                            choices=["picard", "newton"], help="nonlinear solver")
        parser.add_argument("--ls", type=str, dest="pcdls", default="direct",
                            choices=["direct", "iterative"], help="PCD linear solvers")
        parser.add_argument("--ksp_monitor", action="store_true")
        self.args = parser.parse_args(argv)

    def mesh(self, comm):
        mesh = Mesh(comm, "mesh/mesh.xml.gz")
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", triangle, 2)
        Qe = FiniteElement("CG", triangle, 1)
        Ze = MixedElement([Ve, Qe])
        Z  = FunctionSpace(mesh, Ze)
        return Z

    def parameters(self):
        Re = Constant(0)
        return [(Re, "Re", r"$\mathrm{Re}$")]

    def residual(self, z, params, w):
        (u, p) = split(z)
        (v, q) = split(w)

        Re = params[0]
        mesh = z.function_space().mesh()

        # Variational form
        F = (
              1.0/Re * inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - p*div(v)*dx
            - q*div(u)*dx
            )

        # Trial functions
        u_, p_ = TrialFunctions(z.function_space())

        # Jacobian
        if self.args.nls == "picard":
            J = (
                  1.0/Re * inner(grad(u_), grad(v))*dx
                + inner(grad(u_)*u, v)*dx
                - p_*div(v)*dx
                - q*div(u_)*dx
            )
        elif self.args.nls == "newton":
            J = derivative(F, z)

        # Store for later use
        self._J = J

        # If not using PCD we are done
        if self.args.pcd == "none":
            return F

        if self.args.pcdls == "iterative":
            # Add stabilization for AMG 00-block
            iRe = Expression("1./Re", Re=Re, degree=0, mpi_comm=mesh.mpi_comm())
            delta = StabilizationParameterSD(z.sub(0), iRe)
            J_pc = J + delta*inner(grad(u_)*u, grad(v)*u)*dx
            J_pc = None
        else:
            J_pc = None

        # Fetch problem BCs and facet markers
        bcs = self.boundary_conditions(z.function_space(), self.parameters())
        colours = self._colours

        # PCD operators
        mp = Re*p_*q*dx
        kp = Re*dot(grad(p_), u)*q*dx
        ap = inner(grad(p_), grad(q))*dx
        if self.args.pcd == "BRM2":
            n = FacetNormal(mesh)
            ds = Measure("ds", subdomain_data=colours)
            kp -= Re*dot(u, n)*p_*q*ds(1)
            #kp -= Re*dot(u, n)*p_*q*ds(0)  # TODO: Is this beneficial?

        # Artificial BC for PCD preconditioner
        if self.args.pcd == "BRM1":
            bc_pcd = DirichletBC(z.function_space().sub(1), 0.0, colours, 1)
        elif self.args.pcd == "BRM2":
            bc_pcd = DirichletBC(z.function_space().sub(1), 0.0, colours, 2)

        # Store what needed for later
        self._pcd_assembler = PCDAssembler(J, F, bcs, J_pc,
                                           ap=ap, kp=kp, mp=mp,
                                           bcs_pcd=bc_pcd)

        return F

    def boundary_conditions(self, Z, params):
        comm = Z.mesh().mpi_comm()

        # Facet markers
        colours = MeshFunction("size_t", Z.mesh(), Z.mesh().topology().dim()-1)
        walls = CompiledSubDomain("on_boundary", mpi_comm=comm)
        inflow = CompiledSubDomain("on_boundary && near(x[0], 0.0)", mpi_comm=comm)
        outflow = CompiledSubDomain("on_boundary && near(x[0], 150.0)", mpi_comm=comm)
        colours.set_all(999)      # interior facets
        walls.mark(colours, 0)    # no-slip facets
        inflow.mark(colours, 1)   # inlet facets
        outflow.mark(colours, 2)  # outlet facets

        # BCs
        poiseuille = Expression(("-(x[1] + 1) * (x[1] - 1)", "0.0"), degree=2, mpi_comm=comm)
        bc_inflow = DirichletBC(Z.sub(0), poiseuille, colours, 1)
        bc_wall = DirichletBC(Z.sub(0), (0, 0), colours, 0)

        bcs = [bc_inflow, bc_wall]

        # Store markers for later use
        self._colours = colours

        return bcs

    def functionals(self):
        def sqL2(z, params):
            (u, p) = split(z)
            # FIXME: Why is here |z|^2
            j = assemble(inner(z, z)*dx)
            return j

        return [(sqL2, "sqL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return Function(Z)

    def number_solutions(self, params):
        return float("inf")

        #Re = params[0]
        #if   Re < 18:  return 1
        #elif Re < 41:  return 3
        #elif Re < 75:  return 5
        #elif Re < 100: return 8
        #else:          return float("inf")

    def solver_parameters(self, params, klass, averaging=False):
        opts = {
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_max_it": 8,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
        }

        if averaging:
            opts.update({
                "snes_max_it": 32,
                "snes_linesearch_damping": 0.1,
            })

        if self.args.pcd == "none":
            # Completely direct solver, no PCD
            opts.update({
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_package": "mumps",
            })
        else:
            # GMRES with PCD
            opts.update({
                "ksp_converged_reason": None,
                "ksp_max_it": 128,
                "ksp_rtol": 1.0e-5,
                "ksp_atol": 0.0,
                "snes_max_linear_solve_fail": 4,
                "ksp_gmres_restart": 128,
                "fieldsplit_p_pc_python_type": "fenapack.PCDPC_"+self.args.pcd,
            })

            if self.args.ksp_monitor:
                opts.update({"ksp_monitor": None})

            if self.args.pcdls == "iterative":
                # Iterative inner PCD solves
                opts.update({
                    "snes_max_it": 16,  # FIXME: Can we improve this?
                    "fieldsplit_u_ksp_type": "richardson",
                    "fieldsplit_u_ksp_max_it": 1,
                    "fieldsplit_u_pc_type": "hypre",
                    "fieldsplit_u_pc_hypre_type": "boomeramg",
                    "fieldsplit_p_PCD_Ap_ksp_type": "richardson",
                    "fieldsplit_p_PCD_Ap_ksp_max_it": 2,
                    "fieldsplit_p_PCD_Ap_pc_type": "hypre",
                    "fieldsplit_p_PCD_Ap_pc_hypre_type": "boomeramg",
                    "fieldsplit_p_PCD_Mp_ksp_type": "chebyshev",
                    "fieldsplit_p_PCD_Mp_ksp_max_it": 5,
                    # FIXME: Only valid in 2D (see SEW book)?
                    "fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues": "0.5, 2.0",
                    "fieldsplit_p_PCD_Mp_pc_type": "jacobi",
                })

        return opts

    def jacobian(self, F, state, params, test, trial):
        return self._J

    def solver(self, problem, params, solver_params, prefix="", **kwargs):
        # Create nonlinear solver
        solver = SNUFLSolver(problem, prefix=prefix,
                             solver_parameters=solver_params,
                             **kwargs)

        # This is enough for completely direct solve without PCD
        if self.args.pcd == "none":
            return solver

        # Create GMRES KSP with PCD
        ksp = PCDKSP(comm=problem.u.function_space().mesh().mpi_comm())

        # Switch ksp and reuse operators
        oldksp = solver.snes.ksp
        ksp.setOperators(*oldksp.getOperators())
        ksp.setOptionsPrefix(oldksp.getOptionsPrefix())
        solver.snes.ksp = ksp
        solver.snes.setFromOptions()

        # Initilize PCD (only possible with assembled operators)
        # NOTE: It is important that we call jacobian() with x, not a dummy
        #       vector, so that x is updated correctly for continuation
        x = as_backend_type(problem.u.vector()).vec()
        # FIXME: Make sure this works when J_pc is not used
        solver.jacobian(solver.snes, x, *solver.snes.ksp.getOperators())
        ksp.init_pcd(self._pcd_assembler)

        return solver


if __name__ == "__main__":
    dc = DeflatedContinuation(problem=NavierStokesProblem(), teamsize=1, verbose=True)
    #dc.run(values={"Re": linspace(10.0, 100.0, 181)})
    dc.run(values={"Re": arange(18.0, 20.5, 0.5)})

    # FIXME: This is not what we want possibly; an average over WORLD,
    #        thus biased by master thread
    list_timings(TimingClear.keep, [TimingType.wall, TimingType.user])

    dc.bifurcation_diagram("sqL2")
    pyplot.title(r"Bifurcation diagram for sudden expansion in a channel")
    pyplot.savefig("bifurcation.pdf")
