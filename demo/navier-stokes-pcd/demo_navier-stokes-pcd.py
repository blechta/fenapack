"""Flow over a backward-facing step. Incompressible Navier-Stokes equations are
solved using Newton/Picard iterative method. Linear solver is based on field
split PCD preconditioning."""

# Copyright (C) 2015-2017 Martin Rehor, Jan Blechta
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
from matplotlib import pyplot

from fenapack import PCDKrylovSolver
from fenapack import PCDAssembler
from fenapack import PCDNewtonSolver, PCDNonlinearProblem
from fenapack import StabilizationParameterSD

import argparse, sys, os

# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=4,
                    help="level of mesh refinement")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
parser.add_argument("--pcd", type=str, dest="pcd_variant", default="BRM2",
                    choices=["BRM1", "BRM2"], help="PCD variant")
parser.add_argument("--nls", type=str, dest="nls", default="picard",
                    choices=["picard", "newton"], help="nonlinear solver")
parser.add_argument("--ls", type=str, dest="ls", default="iterative",
                    choices=["direct", "iterative"], help="linear solvers")
parser.add_argument("--dm", action='store_true', dest="mumps_debug",
                    help="debug MUMPS")
args = parser.parse_args(sys.argv[1:])

# Load mesh from file and refine uniformly
mesh = Mesh(os.path.join(os.path.pardir, "data", "mesh_lshape.xml"))
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
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
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
if args.pcd_variant == "BRM1":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)
elif args.pcd_variant == "BRM2":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 2)

# Provide some info about the current problem
info("Reynolds number: Re = %g" % (2.0/args.viscosity))
info("Dimension of the function space: %g" % W.dim())

# Arguments and coefficients of the form
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
w = Function(W)
# FIXME: Which split is correct? Both work but one might use
# restrict_as_ufc_function
u_, p_ = split(w)
#u_, p_ = w.split()
nu = Constant(args.viscosity)

# Nonlinear equation
F = (
      nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    - q*div(u_)
)*dx

# Jacobian
if args.nls == "picard":
    J = (
          nu*inner(grad(u), grad(v))
        + inner(dot(grad(u), u_), v)
        - p*div(v)
        - q*div(u)
    )*dx
elif args.nls == "newton":
    J = derivative(F, w)

# Add stabilization for AMG 00-block
if args.ls == "iterative":
    delta = StabilizationParameterSD(w.sub(0), nu)
    J_pc = J + delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx
elif args.ls == "direct":
    J_pc = None

# PCD operators
mp = Constant(1.0/nu)*p*q*dx
kp = Constant(1.0/nu)*dot(grad(p), u_)*q*dx
ap = inner(grad(p), grad(q))*dx
if args.pcd_variant == "BRM2":
    n = FacetNormal(mesh)
    ds = Measure("ds", subdomain_data=boundary_markers)
    kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(1)
    #kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(0)  # TODO: Is this beneficial?

# Collect forms to define nonlinear problem
pcd_assembler = PCDAssembler(J, F, [bc0, bc1],
                             J_pc, ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)
problem = PCDNonlinearProblem(pcd_assembler)

# Set up linear solver (GMRES with right preconditioning using Schur fact)
linear_solver = PCDKrylovSolver(comm=mesh.mpi_comm())
linear_solver.parameters["relative_tolerance"] = 1e-6
PETScOptions.set("ksp_monitor")
PETScOptions.set("ksp_gmres_restart", 150)

# Set up subsolvers
PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDPC_" + args.pcd_variant)
if args.ls == "iterative":
    PETScOptions.set("fieldsplit_u_ksp_type", "richardson")
    PETScOptions.set("fieldsplit_u_ksp_max_it", 1)
    PETScOptions.set("fieldsplit_u_pc_type", "hypre")
    PETScOptions.set("fieldsplit_u_pc_hypre_type", "boomeramg")
    PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_type", "richardson")
    PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_max_it", 2)
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_type", "hypre")
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")
    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_max_it", 5)
    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.0")
    #PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_esteig", "1,0,0,1")  # FIXME: What does it do?
    PETScOptions.set("fieldsplit_p_PCD_Mp_pc_type", "jacobi")
elif args.ls == "direct" and args.mumps_debug:
    # Debugging MUMPS
    PETScOptions.set("fieldsplit_u_mat_mumps_icntl_4", 2)
    PETScOptions.set("fieldsplit_p_PCD_Ap_mat_mumps_icntl_4", 2)
    PETScOptions.set("fieldsplit_p_PCD_Mp_mat_mumps_icntl_4", 2)

# Apply options
linear_solver.set_from_options()

# Set up nonlinear solver
solver = PCDNewtonSolver(linear_solver)
solver.parameters["relative_tolerance"] = 1e-5

# Solve problem
solver.solve(problem, w.vector())

# Report timings
list_timings(TimingClear.clear, [TimingType.wall, TimingType.user])

# Plot solution
u, p = w.split()
size = MPI.size(mesh.mpi_comm())
rank = MPI.rank(mesh.mpi_comm())
pyplot.figure()
pyplot.subplot(2, 1, 1)
plot(u, title="velocity")
pyplot.subplot(2, 1, 2)
plot(p, title="pressure")
pyplot.savefig("figure_v_p_size{}_rank{}.pdf".format(size, rank))
pyplot.figure()
plot(p, title="pressure", mode="warp")
pyplot.savefig("figure_warp_size{}_rank{}.pdf".format(size, rank))
if "CI" not in os.environ:
    pyplot.show()
