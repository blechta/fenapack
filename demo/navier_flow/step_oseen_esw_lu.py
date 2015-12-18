"""Flow over a backward-facing step. Incompressible Navier-Stokes equations are
solved using Oseen approximation. Field split solver is based on PCD
preconditioning proposed by Elman, Silvester and Wathen. All inner linear
solves are performed by LU solver."""

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
from fenapack import FieldSplitSolver

# Reduce logging in parallel
comm = mpi_comm_world()
rank = MPI.rank(comm)
set_log_level(INFO if rank == 0 else INFO+1)
plotting_enabled = True
if MPI.size(comm) > 1:
    plotting_enabled = False # Disable interactive plotting in parallel

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
# Data
nu = Constant(args.viscosity)
f = Constant((0.0, 0.0))
# Variational forms (Stokes problem)
a = (nu*inner(grad(u), grad(v)) - p*div(v) - q*div(u))*dx
L = inner(f, v)*dx
# Add convective term (Oseen approximation)
u0 = Function(V)
a += inner(dot(grad(u), u0), v)*dx
# Assemble system of linear equations
A, b = assemble_system(a, L, bcs)

# Define variational forms for PCD preconditioner
mp = p*q*dx
ap = inner(grad(p), grad(q))*dx
fp = (nu*inner(grad(p), grad(q)) + dot(grad(p), u0)*q)*dx
# Correction of fp due to Robin BC
n = FacetNormal(mesh) # outward unit normal
ds = Measure("ds")[boundary_markers]
fp -= (inner(u0, n)*p*q)*ds(1)
# Assemble PCD operators
Mp = assemble(mp)
Ap = assemble(ap)
Fp = assemble(fp)
# NOTE: Fp changes in every step of nonlinear iteration,
#       while Ap and Mp remain constant.

# Set up field split solver
solver = FieldSplitSolver(W, "gmres")
solver.parameters["monitor_convergence"] = True
solver.parameters["relative_tolerance"] = 1e-6
solver.parameters["maximum_iterations"] = 100
solver.parameters["nonzero_initial_guess"] = True
solver.parameters["error_on_nonconvergence"] = False
solver.parameters["gmres"]["restart"] = 100
# Preconditioner options
pc_prm = solver.parameters["preconditioner"]
pc_prm["side"] = "right"
pc_prm["fieldsplit"]["type"] = "schur"
pc_prm["fieldsplit"]["schur"]["fact_type"] = "upper"

# Set up subsolvers
OptDB_00, OptDB_11 = solver.get_subopts()
# Approximation of 00-block inverse
OptDB_00["ksp_type"] = "preonly"
OptDB_00["pc_type"] = "lu"
#OptDB_00["pc_factor_mat_solver_package"] = "mumps"
#OptDB_00["pc_factor_mat_solver_package"] = "superlu_dist"
# Approximation of 11-block inverse
OptDB_11["ksp_type"] = "preonly"
OptDB_11["pc_type"] = "python"
OptDB_11["pc_python_type"] = "fenapack.PCDPC_ESW"
# PCD specific options: Ap factorization
OptDB_11["PCD_Ap_ksp_type"] = "preonly"
OptDB_11["PCD_Ap_pc_type"] = "lu"
#OptDB_11["PCD_Ap_pc_factor_mat_solver_package"] = "mumps"
#OptDB_11["PCD_Ap_pc_factor_mat_solver_package"] = "superlu_dist"
# PCD specific options: Mp factorization
OptDB_11["PCD_Mp_ksp_type"] = "preonly"
OptDB_11["PCD_Mp_pc_type"] = "lu"
#OptDB_11["PCD_Mp_pc_factor_mat_solver_package"] = "mumps"
#OptDB_11["PCD_Mp_pc_factor_mat_solver_package"] = "superlu_dist"

# Compute solution using Ossen approximation
timer = Timer("Nonlinear solver (Oseen)")
it = 0
max_its = 50 # safety parameter
w = Function(W)
solver.set_operators(A, A, Ap=Ap, Fp=Fp, Mp=Mp, bcs=bcs_pcd)
del Ap, Mp # Release some memory
timer.start()
solver.solve(w.vector(), b) # solve Stokes system (u0 = 0)
while it <= max_its:
    it += 1
    u0.assign(w.sub(0, deepcopy=True))
    A, b = assemble_system(a, L, bcs)
    Fp = assemble(fp)
    solver.set_operators(A, A, Fp=Fp)
    # Stop when number of iterations is zero (residual is small)
    if solver.solve(w.vector(), b) == 0:
        break
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
if plotting_enabled:
    plot(u, title="velocity")
    plot(p, title="pressure", scale=2.0)
    interactive()
