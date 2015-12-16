"""Flow over a backward-facing step. Incompressible Navier-Stokes equations are
solved using Picard iterative method. Field split inner_solver is based on PCD
preconditioning proposed by Elman, Silvester and Wathen. All inner linear
solves are performed by iterative solvers."""

# Copyright (C) 2015 Martin Rehor
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

from dolfin import *
from fenapack import \
     FieldSplitSolver, NonlinearSolver, NonlinearDiscreteProblem, \
     streamline_diffusion_cpp

# Parse input arguments
import argparse, sys
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=4,
                    help="level of mesh refinement")
parser.add_argument("-s", type=float, dest="stretch", default=1.0,
                    help="parameter specifying grid stretch")
parser.add_argument("-nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
args = parser.parse_args(sys.argv[1:])

# Prepare mesh
mesh = Mesh("../../data/step_domain.xml.gz")
# Refinement
numrefs = args.level - 1
for i in range(numrefs):
    mesh = refine(mesh)
# Stretching
if args.stretch != 1.0:
    import numpy as np
    transform_y = lambda y, alpha: np.sign(y)*(abs(y)**alpha)
    x, y = mesh.coordinates().transpose()
    y[:] = transform_y(y, args.stretch)
    it = 0
    for xi in x:
        if xi <= 0.0:
            x[it] = -1.0*abs(xi)**args.stretch
        else:
            x[it] = 5.0*(0.2*xi)**args.stretch
        it += 1
    del it

# Define function spaces (Taylor-Hood)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
W = MixedFunctionSpace((V, Q))

# Define boundary conditions
class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        flag = near(x[1], 0.0) and x[0] <= 0.0
        flag = flag or (near(x[0], 0.0) and x[1] <= 0.0)
        flag = flag or near(x[1], -1.0)
        flag = flag or near(x[1],  1.0)
        return flag and on_boundary
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -1.0) and on_boundary
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 5.0) and on_boundary
# Mark boundaries
boundary_markers = FacetFunction("size_t", mesh)
boundary_markers.set_all(3)
Gamma0().mark(boundary_markers, 0)
Gamma1().mark(boundary_markers, 1)
Gamma2().mark(boundary_markers, 2)
# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, boundary_markers, 0)
# Inflow boundary condition for velocity
inflow = Expression(("4.0*x[1]*(1.0 - x[1])", "0.0"))
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)
# Artificial boundary condition for PCD preconditioning
zero = Constant(0.0)
bc2 = DirichletBC(W.sub(1), zero, boundary_markers, 2)
# Collect boundary conditions
bcs = [bc0, bc1]
bcs_pcd = [bc2]

# Provide some info about the current problem
Re = 2.0/args.viscosity # Reynolds number
info("Reynolds number: Re = %g" % Re)
info("Dimension of the function space: %g" % W.dim())

# Define variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
# Solution vector
w = Function(W)
u_, p_ = split(w)
# Data
nu = Constant(args.viscosity)
f = Constant((0.0, 0.0))
# Nonlinear residual form
F = (
      nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    - q*div(u_)
    - inner(f, v)
)*dx
# Picard correction
J = (
      nu*inner(grad(u), grad(v))
    + inner(dot(grad(u), u_), v)
    - p*div(v)
    - q*div(u)
)*dx
# Preconditioner operator
J_pc = J
# J_pc = (
#       nu*inner(grad(u), grad(v))
#     + inner(dot(grad(u), u_), v)
#     - p*div(v)
#     + Constant(0.0)*p*q
# )*dx
# Add stabilization (streamline diffusion)
delta = Expression(streamline_diffusion_cpp)
delta.nu = args.viscosity
delta.mesh = mesh
delta.wind = w.sub(0, deepcopy=False)
J_pc += delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx

# Define variational forms for PCD preconditioner
mp = p*q*dx
ap = inner(grad(p), grad(q))*dx
fp = (nu*inner(grad(p), grad(q)) + dot(grad(p), u_)*q)*dx
# Correction of fp due to Robin BC
n = FacetNormal(mesh) # outward unit normal
ds = Measure("ds")[boundary_markers]
fp -= (inner(u_, n)*p*q)*ds(1)

# Set up field split inner_solver
inner_solver = FieldSplitSolver(W, "gmres")
inner_solver.parameters["monitor_convergence"] = True
inner_solver.parameters["relative_tolerance"] = 1e-6
inner_solver.parameters["maximum_iterations"] = 100
inner_solver.parameters["error_on_nonconvergence"] = False
inner_solver.parameters["gmres"]["restart"] = 100
# Preconditioner options
pc_prm = inner_solver.parameters["preconditioner"]
pc_prm["side"] = "right"
pc_prm["fieldsplit"]["type"] = "schur"
pc_prm["fieldsplit"]["schur"]["fact_type"] = "upper"
pc_prm["fieldsplit"]["schur"]["precondition"] = "user"

# Set up subsolvers
OptDB_00, OptDB_11 = inner_solver.get_subopts()
# Approximation of 00-block inverse
OptDB_00["ksp_type"] = "richardson"
OptDB_00["pc_type"] = "hypre"
OptDB_00["ksp_max_it"] = 1
OptDB_00["pc_hypre_type"] = "boomeramg"
#OptDB_00["pc_hypre_boomeramg_cycle_type"] = "W"
# Approximation of 11-block inverse
OptDB_11["ksp_type"] = "preonly"
OptDB_11["pc_type"] = "python"
OptDB_11["pc_python_type"] = "fenapack.PCDPC_ESW"
# PCD specific options: Ap factorization
OptDB_11["PCD_Ap_ksp_type"] = "richardson"
OptDB_11["PCD_Ap_pc_type"] = "hypre"
OptDB_11["PCD_Ap_ksp_max_it"] = 2
OptDB_11["PCD_Ap_pc_hypre_type"] = "boomeramg"
# PCD specific options: Mp factorization
OptDB_11["PCD_Mp_ksp_type"] = "chebyshev"
OptDB_11["PCD_Mp_pc_type"] = "jacobi"
OptDB_11["PCD_Mp_ksp_max_it"] = 5
OptDB_11["PCD_Mp_ksp_chebyshev_eigenvalues"] = "0.5, 2.0"
# NOTE: The above estimate is valid for P1 pressure approximation in 2D.

# Define nonlinear problem and initialize linear solver
problem = NonlinearDiscreteProblem(F, J, bcs, J_pc)
A = PETScMatrix()
P = PETScMatrix()
problem.J(A, w.vector())
problem.J_pc(P, w.vector())
inner_solver.set_operators(
    A, P, Ap=assemble(ap), Fp=assemble(fp), Mp=assemble(mp), bcs=bcs_pcd)

# Define hook executed at every nonlinear step
def update_operators(problem, x):
    # Update Jacobian form and PCD operators
    problem.J(A, x)
    problem.J_pc(P, x)
    inner_solver.set_operators(A, P, Fp=assemble(fp))
    if get_log_level() <= PROGRESS:
        plot(delta, mesh=mesh, title="stabilization parameter delta")

# Define nonlinear solver
solver = NonlinearSolver(inner_solver, update_operators)
#solver.parameters["absolute_tolerance"] = 1e-10
solver.parameters["relative_tolerance"] = 1e-5
solver.parameters["maximum_iterations"] = 25
solver.parameters["error_on_nonconvergence"] = False
#solver.parameters["convergence_criterion"] = "incremental"
#solver.parameters["relaxation_parameter"] = 0.5
#solver.parameters["report"] = False

# Compute solution
timer = Timer("Nonlinear solver (Picard)")
timer.start()
solver.solve(problem, w.vector())
timer.stop()

# Split the mixed solution using a shallow copy
u, p = w.split()

# Save solution in XDMF format
filename = sys.argv[0][:-3]
File("results/%s_velocity.xdmf" % filename) << u
File("results/%s_pressure.xdmf" % filename) << p

# Print summary of timings
info("")
list_timings(TimingClear_keep, [TimingType_wall])

# Plot solution
plot(u, title="velocity")
plot(p, title="pressure", scale=2.0)
interactive()
