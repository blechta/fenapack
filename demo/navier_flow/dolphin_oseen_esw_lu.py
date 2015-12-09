# Copyright (C) 2007 Kristian B. Oelgaard, 2008-2009 Anders Logg
# Copyright (C) 2014 Jan Blechta
#
# This file is part of FENaPack and is based on file from DOLFIN.
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
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Martin Rehor 2015

from dolfin import *
from fenapack import FieldSplitSolver

# Parse input arguments
import sys
try:
    viscosity = float(sys.argv[1])
except IndexError:
    viscosity = 0.02

# Load mesh and subdomains
mesh = Mesh("../../data/dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "../../data/dolfin_fine_subdomains.xml.gz")

# # Refine
# # NOTE: Only works sequentially
# num_refinements = 0
# sub_domains_old = []
# for i in range(num_refinements):
#     mesh = adapt(mesh, CellFunction('bool', mesh, True))
#     # BUG in DOLFIN: this is needed to prevent memory corruption
#     sub_domains_old.append(sub_domains)
#     sub_domains = adapt(sub_domains, mesh)

# Define function spaces (Taylor-Hood)
V = VectorFunctionSpace(mesh, 'Lagrange', 2)
Q = FunctionSpace(mesh, 'Lagrange', 1)
W = MixedFunctionSpace((V, Q))
info('Dimension of the function space %g' % W.dim())

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

# Boundary conditions for PCD preconditioning
zero = Constant(0.0)
bc2 = DirichletBC(W.sub(1), zero, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1]
bcs_pcd = [bc2]

# Define variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
nu = Constant(viscosity)
#nu = Expression('x[1] < x1 ? nu1 : nu2', x1=0.5, nu1=1e1, nu2=2e-1)
f  = Constant((0.0, 0.0))
a  = (nu*inner(grad(u), grad(v)) - p*div(v) - q*div(u))*dx
L  = inner(f, v)*dx

# Add Oseen-like convection
u0 = Function(V)
a += inner(dot(grad(u), u0), v)*dx

# Assemble system of linear equations
A, b = assemble_system(a, L, bcs)

# Define operators for PCD preconditioner
mp = p*q*dx
ap = inner(grad(p), grad(q))*dx
fp = (nu*inner(grad(p), grad(q)) + dot(grad(p), u0)*q)*dx
# Correction of fp due to Robin BC
n = FacetNormal(mesh) # outward unit normal
ds = Measure("ds")[sub_domains]
fp -= (inner(u0, n)*p*q)*ds(1)
# Assemble PCD operators
Mp = assemble(mp)
Ap = assemble(ap)
Fp = assemble(fp)

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
# Approximation of 11-block inverse
OptDB_11["ksp_type"] = "preonly"
OptDB_11["pc_type"] = "python"
OptDB_11["pc_python_type"] = "fenapack.PCDPC_ESW"
# PCD specific options: Ap factorization
OptDB_11["PCD_Ap_ksp_type"] = "preonly"
OptDB_11["PCD_Ap_pc_type"] = "lu"
# PCD specific options: Mp factorization
OptDB_11["PCD_Mp_ksp_type"] = "preonly"
OptDB_11["PCD_Mp_pc_type"] = "lu"

# Compute solution using Ossen approximation
it = 0
max_its = 50 # safety parameter
w = Function(W)
solver.set_operator(A)
solver.custom_setup(Ap, Fp, Mp, bcs_pcd)
solver.solve(w.vector(), b) # solve Stokes system (u0 = 0)
while it <= max_its:
    it += 1
    u0.assign(w.sub(0, deepcopy=True))
    A, b = assemble_system(a, L, bcs)
    solver.set_operator(A)
    Fp = assemble(fp)
    solver.custom_setup(Ap, Fp, Mp, bcs_pcd)
    # Stop when number of iterations is zero (residual is small)
    if solver.solve(w.vector(), b) == 0:
        break

# Split the mixed solution using a shallow copy
u, p = w.split()

# Save solution in XDMF format
filename = sys.argv[0][:-3]
File("results/%s_velocity.xdmf" % filename) << u
File("results/%s_pressure.xdmf" % filename) << p

# Plot solution
plot(u, title="velocity")
plot(p, title="pressure", scale=2.0)
interactive()
