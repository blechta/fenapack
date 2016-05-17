"""3D flow over a backward-facing step without the inlet part of the channel.
Incompressible Navier-Stokes equations are solved using Newton/Picard iterative
method. Field split inner solver is based on PCD preconditioning."""

# Copyright (C) 2016 Martin Rehor
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
     StabilizationParameterSD

# Adjust DOLFIN's global parameters
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True

# Reduce logging in parallel
comm = mpi_comm_world()
rank = MPI.rank(comm)
set_log_level(PROGRESS if rank == 0 else INFO+1)
plotting_enabled = True
if MPI.size(comm) > 1:
    plotting_enabled = False # Disable interactive plotting in parallel

# Parse input arguments
import argparse, sys
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=3,
                    help="level of mesh refinement")
parser.add_argument("-s", type=float, dest="stretch", default=1.0,
                    help="parameter specifying grid stretch")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
parser.add_argument("--nls", type=str, dest="nls",
                    choices=["Newton", "Picard"], default="Picard",
                    help="type of nonlinear solver")
parser.add_argument("--PCD", type=str, dest="pcd_strategy",
                    choices=["BRM", "SEW"], default="SEW",
                    help="strategy used for PCD preconditioning")
parser.add_argument("--insolver", type=str, dest="insolver",
                    choices=["lu", "it"], default="lu",
                    help="direct or iterative inner solver")
parser.add_argument("--save", action="store_true", dest="save_results",
                    help="save results")
args = parser.parse_args(sys.argv[1:])

# Prepare mesh
channel_length = 5.0
bot_corner = Point(0.0, -1.0, 0.0)
top_corner = Point(channel_length, 1.0, 1.0)
mesh = BoxMesh(comm, bot_corner, top_corner,
               int(channel_length), 2, 1)
# Refinement
numrefs = args.level - 1
for i in range(numrefs):
    mesh = refine(mesh)
# Stretching
if args.stretch != 1.0:
    import numpy as np
    transform_y = lambda y, alpha: np.sign(y)*(abs(y)**alpha)
    x, y, z = mesh.coordinates().transpose()
    y[:] = transform_y(y, args.stretch)
    it = 0
    for xi in x:
        x[it] = channel_length*(1.0/channel_length*xi)**args.stretch
        it += 1
    del it

# Define function space (Taylor-Hood)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)

# Define boundary conditions
class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        flag = near(x[0], 0.0) and x[1] <= 0.0
        flag = flag or near(x[1], -1.0)
        flag = flag or near(x[1],  1.0)
        return flag and on_boundary
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and x[1] >= 0.0 and on_boundary
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], channel_length) and on_boundary
class Gamma3(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[2],  0.0) or near(x[2],  1.0)) and on_boundary
# Mark boundaries
boundary_markers = FacetFunction("size_t", mesh)
boundary_markers.set_all(4)
Gamma0().mark(boundary_markers, 0)
Gamma1().mark(boundary_markers, 1)
Gamma2().mark(boundary_markers, 2)
Gamma3().mark(boundary_markers, 3)
# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, boundary_markers, 0)
# Inflow boundary condition for velocity
inflow = Expression(("4.0*x[1]*(1.0 - x[1])", "0.0", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)
# Full slip boundary condition
zero = Constant(0.0)
bc3 = DirichletBC(W.sub(0).sub(2), zero, boundary_markers, 3)
# Artificial boundary condition for PCD preconditioning
PCD_BND_MARKER = 2 if args.pcd_strategy == "SEW" else 1
bc_art = DirichletBC(W.sub(1), zero, boundary_markers, PCD_BND_MARKER)
# Collect boundary conditions
bcs = [bc0, bc1, bc3]
bcs_pcd = [bc_art]

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
n = FacetNormal(mesh) # outward unit normal
ds = Measure("ds", subdomain_data=boundary_markers)
nu = Constant(args.viscosity)
inu = Constant(1.0/args.viscosity)
f = Constant((0.0, 0.0, 0.0))
F = (
      nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    - q*div(u_)
    - inner(f, v)
)*dx
# Newton/Picard correction
if args.nls == "Newton":
    J = derivative(F, w)
else:
    J = (
          nu*inner(grad(u), grad(v))
        + inner(dot(grad(u), u_), v)
        - p*div(v)
        - q*div(u)
    )*dx
# Preconditioner
if args.insolver == "it":
    J_pc = (
          nu*inner(grad(u), grad(v))
        + inner(dot(grad(u), u_), v)
        - p*div(v)
        - inu*p*q # this term is irrelevant when using PCD
    )*dx
    # Add stabilization (streamline diffusion) to preconditioner
    delta = StabilizationParameterSD(w.sub(0), nu)
    J_pc += delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx

# Define variational forms for PCD preconditioner
mp = p*q*dx
kp = dot(grad(p), u_)*q*dx
ap = inner(grad(p), grad(q))*dx
if args.pcd_strategy == "BRM":
    mp = inu*mp
    kp = inu*kp
fp = nu*ap + kp

# Collect forms to define nonlinear problem
problem_args = [F, bcs, J]
problem_args += [J_pc] if args.insolver == "it" else []
if args.pcd_strategy == "SEW":
    fp -= (inner(u_, n)*p*q)*ds(1) # Correction of fp due to Robin BC
    problem = NonlinearDiscreteProblem(
        *problem_args, ap=ap, fp=fp, mp=mp, bcs_pcd=bcs_pcd)
else:
    problem = NonlinearDiscreteProblem(
        *problem_args, ap=ap, kp=kp, mp=mp, bcs_pcd=bcs_pcd)

# Set up field split inner_solver
inner_solver = FieldSplitSolver(W, "gmres")
inner_solver.parameters["monitor_convergence"] = True
inner_solver.parameters["relative_tolerance"] = 1e-6
inner_solver.parameters["maximum_iterations"] = 100
#inner_solver.parameters["nonzero_initial_guess"] = True
inner_solver.parameters["error_on_nonconvergence"] = False
#inner_solver.parameters["gmres"]["restart"] = 100 # FIXME: Need to set restart through petsc4py
# Preconditioner options
pc_prm = inner_solver.parameters["preconditioner"]
pc_prm["side"] = "right"
pc_prm["fieldsplit"]["type"] = "schur"
pc_prm["fieldsplit"]["schur"]["fact_type"] = "upper"
pc_prm["fieldsplit"]["schur"]["precondition"] = "user"

# Set up subsolvers
OptDB_00, OptDB_11 = inner_solver.get_subopts()
# Approximation of 11-block inverse
OptDB_11["ksp_type"] = "preonly"
OptDB_11["pc_type"] = "python"
OptDB_11["pc_python_type"] = "_".join(["fenapack.PCDPC", args.pcd_strategy])
if args.insolver == "lu":
    # Approximation of 00-block inverse
    OptDB_00["ksp_type"] = "preonly"
    OptDB_00["pc_type"] = "lu"
    OptDB_00["pc_factor_mat_solver_package"] = "mumps"
    # PCD specific options: Ap factorization
    OptDB_11["PCD_Ap_ksp_type"] = "preonly"
    OptDB_11["PCD_Ap_pc_type"] = "lu"
    OptDB_11["PCD_Ap_pc_factor_mat_solver_package"] = "mumps"
    # PCD specific options: Mp factorization
    OptDB_11["PCD_Mp_ksp_type"] = "preonly"
    OptDB_11["PCD_Mp_pc_type"] = "lu"
    OptDB_11["PCD_Mp_pc_factor_mat_solver_package"] = "mumps"
else:
    # Approximation of 00-block inverse
    OptDB_00["ksp_type"] = "richardson"
    OptDB_00["pc_type"] = "hypre"
    OptDB_00["ksp_max_it"] = 1
    OptDB_00["pc_hypre_type"] = "boomeramg"
    #OptDB_00["pc_hypre_boomeramg_coarsen_type"] = "Falgout"
    # possibilities: CLJP Ruge-Stueben modifiedRuge-Stueben PMIS PMIS HMIS
    OptDB_00["pc_hypre_boomeramg_interp_type"] = "multipass"
    # possibilities: multipass ext+i-cc FF1 FF classical ext+i direct
    #                multipass-wts standard standard-wts
    # PCD specific options: Ap factorization
    OptDB_11["PCD_Ap_ksp_type"] = "richardson"
    OptDB_11["PCD_Ap_pc_type"] = "hypre"
    OptDB_11["PCD_Ap_ksp_max_it"] = 2
    OptDB_11["PCD_Ap_pc_hypre_type"] = "boomeramg"
    # PCD specific options: Mp factorization
    OptDB_11["PCD_Mp_ksp_type"] = "chebyshev"
    OptDB_11["PCD_Mp_pc_type"] = "jacobi"
    OptDB_11["PCD_Mp_ksp_max_it"] = 5
    OptDB_11["PCD_Mp_ksp_chebyshev_eigenvalues"] = "0.5, 2.5"
    # NOTE: The above estimate is valid for P1 pressure approximation in 3D.

# Define debugging hook executed at every nonlinear step
def plot_delta(*args, **kwargs):
    if plotting_enabled and get_log_level() <= PROGRESS:
        plot(delta, mesh=mesh, title="stabilization parameter delta")

# Set up nonlinear solver
hook = plot_delta if args.insolver == "it" else None
solver = NonlinearSolver(inner_solver, debug_hook=hook)
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
u.rename("v", "velocity")
p.rename("p", "pressure")

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
