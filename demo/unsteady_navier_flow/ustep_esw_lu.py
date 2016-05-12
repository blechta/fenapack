"""Unsteady flow over a backward-facing step. Incompressible unsteady
Navier-Stokes equations are solved using fully implicit schemes for time
discretization of the equations. Field split solver for linearized problems is
based on PCD preconditioning proposed by Elman, Silvester and Wathen. Inner
linear solves are performed by LU solver."""

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
from fenapack import FieldSplitSolver, NonlinearSolver, NonlinearDiscreteProblem

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
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
parser.add_argument("--dt", type=float, dest="dt", default=None,
                    help="constant time step")
parser.add_argument("--te", type=float, dest="t_end", default=10.0,
                    help="final time of the simulation")
parser.add_argument("--nls", type=str, dest="nls",
                    choices=["Newton", "Picard"], default="Picard",
                    help="type of nonlinear solver")
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

# Define function space (Taylor-Hood)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)

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
inflow = Expression(("(1.0 - exp(-5.0*t))*4.0*x[1]*(1.0 - x[1])", "0.0"), t=0.0,
                    degree=2)
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)
# Artificial boundary condition for PCD preconditioning
zero = Constant(0.0)
bc2 = DirichletBC(W.sub(1), zero, boundary_markers, 2)
# Collect boundary conditions
bcs = [bc0, bc1]
bcs_pcd = [bc2]

# Get time step
umax = 1.0 # umax = inflow((-1.0, 0.5))[0]
h = 1.0/(2.0**numrefs)
#h = mesh.hmin()
if not args.dt:
    dt = h/umax
else:
    dt = args.dt
CFLin = dt*umax/h # Courant number near inlet (if args.stretch == 1)
T_END = args.t_end

# Provide some info about the current problem
Re = 2.0/args.viscosity # Reynolds number
info("Reynolds number: Re = %g" % Re)
info("Dimension of the function space: %g" % W.dim())
info("Time step: dt = %g (CFL near inlet ... %g)" % (dt, CFLin))

# Define variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
# Solution vector
w = Function(W)
u_, p_ = split(w)
# Solution vector at previous time step
w0 = Function(W)
u_0, p_0 = split(w0)
# Solution vector at last but one time step
w1 = Function(W)
u_1, p_1 = split(w1)
# Data
k = Constant(dt)
ik = Constant(1.0/dt)
nu = Constant(args.viscosity)
f = Constant((0.0, 0.0))
# -----------------------------------------------------------------------------
# Define nonlinear backward Euler scheme
F_BE = (
      ik*inner(u_-u_0, v)
    + inner(dot(grad(u_), u_), v)
    + nu*inner(grad(u_), grad(v))
    - p_*div(v)
    - q*div(u_)
    - inner(f, v)
)*dx
# Newton or Picard correction
if args.nls == "Newton":
    J_BE = derivative(F_BE, w)
else:
    J_BE = (
          ik*inner(u, v)
        + inner(dot(grad(u), u_), v)
        + nu*inner(grad(u), grad(v))
        - p*div(v)
        - q*div(u)
    )*dx
# Variational forms for PCD preconditioner
mu = inner(u, v)*dx
mp = p*q*dx
ap = inner(grad(p), grad(q))*dx
fp_BE = (
      ik*p*q
    + dot(grad(p), u_)*q
    + nu*inner(grad(p), grad(q))
)*dx
n = FacetNormal(mesh) # outward unit normal
ds = Measure("ds", subdomain_data=boundary_markers)
fp_BE -= (inner(u_, n)*p*q)*ds(1) # correction of fp due to Robin BC
# Set up inner solver
inner_solver_BE = FieldSplitSolver(W, "gmres", "solver_BE_")
inner_solver_BE.parameters["monitor_convergence"] = True
inner_solver_BE.parameters["relative_tolerance"] = 1e-6
inner_solver_BE.parameters["maximum_iterations"] = 100
#inner_solver_BE.parameters["nonzero_initial_guess"] = True
#inner_solver_BE.parameters["error_on_nonconvergence"] = False
#inner_solver_BE.parameters["gmres"]["restart"] = 100 # FIXME: Need to set restart through petsc4py
pc_prm = inner_solver_BE.parameters["preconditioner"]
pc_prm["side"] = "right"
pc_prm["fieldsplit"]["type"] = "schur"
pc_prm["fieldsplit"]["schur"]["fact_type"] = "upper"
pc_prm["fieldsplit"]["schur"]["precondition"] = "user"
# Set up subsolvers
OptDB_00, OptDB_11 = inner_solver_BE.get_subopts()
# Approximation of 00-block inverse
OptDB_00["ksp_type"] = "preonly"
OptDB_00["pc_type"] = "lu"
OptDB_00["pc_factor_mat_solver_package"] = "mumps"
# Approximation of 11-block inverse
OptDB_11["ksp_type"] = "preonly"
OptDB_11["pc_type"] = "python"
OptDB_11["pc_python_type"] = "fenapack.UnsteadyPCDPC_ESW"
# PCD specific options: Ap factorization
OptDB_11["PCD_Ap_ksp_type"] = "preonly"
OptDB_11["PCD_Ap_pc_type"] = "lu"
OptDB_11["PCD_Ap_pc_factor_mat_solver_package"] = "mumps"
# PCD specific options: Mp factorization
OptDB_11["PCD_Mp_ksp_type"] = "preonly"
OptDB_11["PCD_Mp_pc_type"] = "lu"
OptDB_11["PCD_Mp_pc_factor_mat_solver_package"] = "mumps"
# Nonlinear problem and solver
problem_BE = NonlinearDiscreteProblem(
    F_BE, bcs, J_BE, mu=mu, fp=fp_BE, mp=mp, bcs_pcd=bcs_pcd)
solver_BE = NonlinearSolver(inner_solver_BE)
#solver_BE.parameters["absolute_tolerance"] = 1e-10
solver_BE.parameters["relative_tolerance"] = 1e-5
solver_BE.parameters["maximum_iterations"] = 25
#solver_BE.parameters["error_on_nonconvergence"] = False
#solver_BE.parameters["convergence_criterion"] = "incremental"
#solver_BE.parameters["relaxation_parameter"] = 0.5
#solver_BE.parameters["report"] = False
# -----------------------------------------------------------------------------
# Define Simo-Armero scheme
theta = 0.5
ctheta1 = Constant(theta)
ctheta2 = Constant(1.0-theta)
u_star = Constant(1.5)*u_0 - Constant(0.5)*u_1
a_SA = (
      ik*inner(u, v)
    + ctheta1*inner(dot(grad(u), u_star), v)
    + ctheta1*nu*inner(grad(u), grad(v))
    - ctheta1*p*div(v)
    - ctheta1*q*div(u)
)*dx
L_SA = (
      inner(f, v)
    + ik*inner(u_0, v)
    - ctheta2*inner(dot(grad(u_0), u_star), v)
    - ctheta2*nu*inner(grad(u_0), grad(v))
    + ctheta2*p_0*div(v)
    + ctheta2*q*div(u_0)
)*dx
# Variational forms for PCD preconditioner
fp_SA = (
      ik*p*q
    + ctheta1*dot(grad(p), u_star)*q
    + ctheta1*nu*inner(grad(p), grad(q))
)*dx
fp_SA -= ctheta1*(inner(u_star, n)*p*q)*ds(1) # correction of fp due to Robin BC
# Set up linear solver
solver_SA = FieldSplitSolver(W, "gmres", "solver_SA_")
solver_SA.parameters["monitor_convergence"] = True
solver_SA.parameters["relative_tolerance"] = 1e-6
solver_SA.parameters["maximum_iterations"] = 100
#solver_SA.parameters["nonzero_initial_guess"] = True
#solver_SA.parameters["error_on_nonconvergence"] = False
#solver_SA.parameters["gmres"]["restart"] = 100 # FIXME: Need to set restart through petsc4py
pc_prm = solver_SA.parameters["preconditioner"]
pc_prm["side"] = "right"
pc_prm["fieldsplit"]["type"] = "schur"
pc_prm["fieldsplit"]["schur"]["fact_type"] = "upper"
pc_prm["fieldsplit"]["schur"]["precondition"] = "user"
# Set up subsolvers
OptDB_00, OptDB_11 = solver_SA.get_subopts()
# Approximation of 00-block inverse
OptDB_00["ksp_type"] = "preonly"
OptDB_00["pc_type"] = "lu"
OptDB_00["pc_factor_mat_solver_package"] = "mumps"
# Approximation of 11-block inverse
OptDB_11["ksp_type"] = "preonly"
OptDB_11["pc_type"] = "python"
OptDB_11["pc_python_type"] = "fenapack.UnsteadyPCDPC_ESW"
# PCD specific options: Ap factorization
OptDB_11["PCD_Ap_ksp_type"] = "preonly"
OptDB_11["PCD_Ap_pc_type"] = "lu"
OptDB_11["PCD_Ap_pc_factor_mat_solver_package"] = "mumps"
# PCD specific options: Mp factorization
OptDB_11["PCD_Mp_ksp_type"] = "preonly"
OptDB_11["PCD_Mp_pc_type"] = "lu"
OptDB_11["PCD_Mp_pc_factor_mat_solver_package"] = "mumps"
# Set operators
A = assemble(a_SA)
for bc in bcs:
    bc.apply(A)
solver_SA.set_operators(
    A, A, Mu=assemble(mu), Fp=assemble(fp_SA), Mp=assemble(mp), bcs=bcs_pcd)
# -----------------------------------------------------------------------------

# Save solution in XDMF format
if args.save_results:
    comm = mpi_comm_world()
    filename = sys.argv[0][:-3]
    vfile = XDMFFile(comm, "results/%s_velocity.xdmf" % filename)
    pfile = XDMFFile(comm, "results/%s_pressure.xdmf" % filename)
    vfile.parameters["rewrite_function_mesh"] = False
    pfile.parameters["rewrite_function_mesh"] = False

# Save and plot initial solution
t = 0.0
u, p = w.split()
u.rename("v", "velocity")
p.rename("p", "pressure")
if args.save_results:
    vfile.write(u, t)
    pfile.write(p, t)
if plotting_enabled:
    plot(u, title="velocity @ t = %g" % t)
    plot(p, title="pressure @ t = %g" % t, scale=2.0)

# Compute solution
timer = Timer("Time stepping")
timer.start()
# Time stepping scheme
while t < T_END:
    # Update time variables
    t += dt
    inflow.t = t
    # Solve linear problem
    begin("t = %g:" % t)
    if t <= 2.0: #*dt:
        info("Performing a single step of backward Euler scheme.")
        solver_BE.solve(problem_BE, w.vector())
    else:
        info("Performing a single step of Simo-Armero scheme.")
        #A, b = assemble_system(a_SA, L_SA, bcs)
        A = assemble(a_SA)
        b = assemble(L_SA)
        for bc in bcs:
            bc.apply(A, b)
        solver_SA.set_operators(A, A, Fp=assemble(fp_SA))
        solver_SA.solve(w.vector(), b)
    end()
    # Update solution variables at previous time steps
    w1.assign(w0)
    w0.assign(w)
    # Save and plot results
    if args.save_results:
        vfile.write(u, t)
        pfile.write(p, t)
    if plotting_enabled:
        plot(u, title="velocity @ t = %g" % t)
        plot(p, title="pressure @ t = %g" % t, scale=2.0)
timer.stop()

# Print summary of timings
info("")
list_timings(TimingClear_keep, [TimingType_wall])

# Plot solution
if plotting_enabled:
    plot(u, title="velocity @ t = %g" % t)
    plot(p, title="pressure @ t = %g" % t, scale=2.0)
    interactive()
