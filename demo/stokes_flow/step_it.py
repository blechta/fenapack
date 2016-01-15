"""Slow flow over a backward-facing step. Stokes equations are solved using
iterative field split solver with block diagonal preconditioning. Inner linear
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
from fenapack import FieldSplitSolver

# Adjust DOLFIN's global parameters
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True

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
parser.add_argument("--fs_type", type=str, dest="fieldsplit_type",
                    choices=["additive", "schur"], default="additive",
                    help="fieldsplit type")
parser.add_argument("--AMG", type=str, dest="AMG",
                    choices=["hypre", "gamg", "ml"], default="hypre",
                    help="type of AMG preconditioner")
parser.add_argument("--MMP", type=str, dest="MMP",
                    choices=["diag", "cheb", "cg"], default="cheb",
                    help="type of mass matrix preconditioner")
parser.add_argument("--save", action="store_true", dest="save_results",
                    help="save results")
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
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))

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
# Collect boundary conditions
bcs = [bc0, bc1]

# Provide some info about the current problem
info("Dimension of the function space: %g" % W.dim())

# Define variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
# Data
f = Constant((0.0, 0.0))
# Variational forms (Stokes problem)
a = (inner(grad(u), grad(v)) - p*div(v) - q*div(u))*dx
L = inner(f, v)*dx
# Assemble system of linear equations
A, b = assemble_system(a, L, bcs)

# Define variational form for block diagonal preconditioner
sgn = -1.0 if args.fieldsplit_type == "schur" else 1.0
a_pc = (inner(grad(u), grad(v)) + Constant(sgn)*p*q)*dx
# NOTE: For explanation of minus sign above see DIAG on manual
#       page of PETSc function 'PCFieldSplitSetSchurFactType'.
# Assemble preconditioner
assembler = SystemAssembler(a_pc, L, bcs)
P = PETScMatrix()
assembler.assemble(P)

# Set up field split solver
solver = FieldSplitSolver(W, "minres")
solver.parameters["monitor_convergence"] = True
solver.parameters["relative_tolerance"] = 1e-6
solver.parameters["maximum_iterations"] = 100
solver.parameters["error_on_nonconvergence"] = False
# Preconditioner options
pc_prm = solver.parameters["preconditioner"]
pc_prm["side"] = "left"
pc_prm["fieldsplit"]["type"] = args.fieldsplit_type
pc_prm["fieldsplit"]["schur"]["fact_type"] = "diag"
pc_prm["fieldsplit"]["schur"]["precondition"] = "a11"

# Set up subsolvers
OptDB_00, OptDB_11 = solver.get_subopts()
# Approximation of 00-block inverse
# TODO: Take a look at the options more carefully.
if args.AMG == "hypre":
    OptDB_00["ksp_type"] = "richardson"
    OptDB_00["ksp_max_it"] = 1
    OptDB_00["pc_type"] = "hypre"
    OptDB_00["pc_hypre_type"] = "boomeramg"
elif args.AMG == "ml":
    OptDB_00["ksp_type"] = "richardson"
    OptDB_00["pc_type"] = "ml"
    OptDB_00["ksp_max_it"] = 1
    OptDB_00["mg_levels_ksp_type"] = "chebyshev"
    OptDB_00["mg_levels_ksp_max_it"] = 2
    OptDB_00["mg_levels_pc_type"] = "sor"
    OptDB_00["mg_levels_pc_sor_its"] = 2
elif args.AMG == "gamg":
    OptDB_00["ksp_type"] = "richardson"
    OptDB_00["pc_type"] = "gamg"
    OptDB_00["ksp_max_it"] = 1
    OptDB_00["mg_levels_ksp_type"] = "chebyshev"
    OptDB_00["mg_levels_ksp_max_it"] = 2
    OptDB_00["mg_levels_pc_type"] = "sor"
    OptDB_00["mg_levels_pc_sor_its"] = 2
# Approximation of 11-block inverse
if args.MMP == "diag":
    OptDB_11["ksp_type"] = "richardson"
    OptDB_11["pc_type"] = "jacobi"
    OptDB_11["ksp_max_it"] = 1
elif args.MMP == "cheb":
    OptDB_11["ksp_type"] = "chebyshev"
    OptDB_11["pc_type"] = "jacobi"
    OptDB_11["ksp_max_it"] = 5
    OptDB_11["ksp_chebyshev_eigenvalues"] = "0.5, 2.0"
    # NOTE: The above estimate is valid for P1 pressure approximation in 2D.
elif args.MMP == "cg":
    OptDB_11["ksp_type"] = "cg"
    OptDB_11["ksp_max_it"] = 5

# Compute solution of the Stokes system
w = Function(W)
solver.set_operators(A, P)
solver.solve(w.vector(), b)

# Split the mixed solution using a shallow copy
u, p = w.split()

# Save solution in XDMF format
if args.save_results:
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
