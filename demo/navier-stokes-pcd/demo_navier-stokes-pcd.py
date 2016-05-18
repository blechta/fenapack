"""Flow over a backward-facing step. Incompressible Navier-Stokes equations are
solved using Newton/Picard iterative method. Linear solver is based on field
split PCD preconditioning."""

# Copyright (C) 2015-2016 Martin Rehor
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

# Begin demo

from dolfin import *
from fenapack import FieldSplitSolver, NewtonSolver, PCDProblem, StabilizationParameterSD


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True

# Parse input arguments
import argparse, sys
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=4,
                    help="level of mesh refinement")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
args = parser.parse_args(sys.argv[1:])

# Load mesh from file and refine uniformly
mesh = Mesh("../../data/step_domain.xml.gz")
for i in range(args.level):
    mesh = refine(mesh)

# Define and mark boundaries
class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -1.0)
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 5.0)
boundary_markers = FacetFunction("size_t", mesh)
boundary_markers.set_all(3)        # interior facets
Gamma0().mark(boundary_markers, 0) # no-slip facets
Gamma1().mark(boundary_markers, 1) # inlet facets
Gamma2().mark(boundary_markers, 2) # outlet facets

# Build Taylor-Hood function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)

# No-slip BC
bc0 = DirichletBC(W.sub(0), (0.0, 0.0), boundary_markers, 0)

# Parabolic inflow BC
inflow = Expression(("4.0*x[1]*(1.0 - x[1])", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)

# Artificial BC for PCD preconditioner
bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)

# Provide some info about the current problem
info("Reynolds number: Re = %g" % (2.0/args.viscosity))
info("Dimension of the function space: %g" % W.dim())

# Arguments and coefficients of the form
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
w = Function(W)
u_, p_ = split(w)
n = FacetNormal(mesh)
nu = Constant(args.viscosity)
ds = Measure("ds", subdomain_data=boundary_markers)

# Nonlinear equation
F = (
      nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    - q*div(u_)
)*dx

# Picard linearization (one could use full Newton)
J = (
      nu*inner(grad(u), grad(v))
    + inner(dot(grad(u), u_), v)
    - p*div(v)
    - q*div(u)
)*dx

# Add stabilization for AMG 00-block
delta = StabilizationParameterSD(w.sub(0), nu)
J_pc = J + delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx

# PCD operators
mp = Constant(1.0/nu)*p*q*dx
kp = Constant(1.0/nu)*dot(grad(p), u_)*q*dx
ap = inner(grad(p), grad(q))*dx

# Collect forms to define nonlinear problem
problem = PCDProblem(F, [bc0, bc1], J, J_pc, ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)

# Set up linear field split solver
linear_solver = FieldSplitSolver(W, "gmres")
linear_solver.parameters["monitor_convergence"] = True
linear_solver.parameters["relative_tolerance"] = 1e-6
linear_solver.parameters["nonzero_initial_guess"] = False
linear_solver.parameters["preconditioner"]["side"] = "right"
linear_solver.parameters["preconditioner"]["fieldsplit"]["type"] = "schur"
linear_solver.parameters["preconditioner"]["fieldsplit"]["schur"]["fact_type"] = "upper"
linear_solver.parameters["preconditioner"]["fieldsplit"]["schur"]["precondition"] = "user"

# Set up subsolvers
OptDB_00, OptDB_11 = linear_solver.get_subopts()
OptDB_00["ksp_type"] = "richardson"
OptDB_00["ksp_max_it"] = 1
OptDB_00["pc_type"] = "hypre"
OptDB_00["pc_hypre_type"] = "boomeramg"
OptDB_11["ksp_type"] = "preonly"
OptDB_11["pc_type"] = "python"
OptDB_11["pc_python_type"] = "fenapack.PCDPC_BRM"
OptDB_11["PCD_Ap_ksp_type"] = "richardson"
OptDB_11["PCD_Ap_ksp_max_it"] = 2
OptDB_11["PCD_Ap_pc_type"] = "hypre"
OptDB_11["PCD_Ap_pc_hypre_type"] = "boomeramg"
OptDB_11["PCD_Mp_ksp_type"] = "chebyshev"
OptDB_11["PCD_Mp_ksp_max_it"] = 5
OptDB_11["PCD_Mp_ksp_chebyshev_eigenvalues"] = "0.5, 2.0"
OptDB_11["PCD_Mp_pc_type"] = "jacobi"

# Set up nonlinear solver
solver = NewtonSolver(linear_solver)
solver.parameters["relative_tolerance"] = 1e-5

# Solve problem
solver.solve(problem, w.vector())

# Plot solution
u, p = w.split()
plot(u, title="velocity")
plot(p, title="pressure")
interactive()
