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
from fenapack import PCDFieldSplitSolver

import sys

try:
    strategy = sys.argv[1]
except IndexError:
    strategy = 'A'

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
if strategy == 'A':
    print "Usage of strategy A."
    bc2 = DirichletBC(W.sub(1), zero, sub_domains, 2)
else:
    print "Usage of strategy B."
    bc2 = DirichletBC(W.sub(1), zero, sub_domains, 1)

# Collect boundary conditions
bcs = [bc0, bc1]
bcs_pcd = [bc2]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
nu = Constant(1e-1)
#nu = Expression('x[1] < x1 ? nu1 : nu2', x1=0.5, nu1=1e1, nu2=2e-1)
f  = Constant((0.0, 0.0))
a  = (nu*inner(grad(u), grad(v)) - p*div(v) - q*div(u))*dx
L  = inner(f, v)*dx

# Add Oseen-like convection
u0 = Function(V)
a += inner(dot(grad(u), u0), v)*dx

# Define outward unit normal
n = FacetNormal(W.mesh())

# Operators for PCD preconditioner
mp = p*q*dx
ap = inner(grad(p), grad(q))*dx
fp = (nu*inner(grad(p), grad(q)) + dot(grad(p), u0)*q)*dx
alpha = 0.0 if strategy == 'B' else 1.0
fp -= Constant(alpha)*inner(u0, n)*p*q*ds # correction due to Robin BC
Lp = Constant(0.0)*q*dx # dummy right-hand side

# Assemble
A, b = assemble_system(a, L, bcs)

# Setup fieldsplit solver
solver = PCDFieldSplitSolver(W, "gmres")
solver.parameters["monitor_convergence"] = True
solver.parameters["relative_tolerance"] = 1e-6
solver.parameters["maximum_iterations"] = 100
solver.parameters["nonzero_initial_guess"] = True
solver.parameters["error_on_nonconvergence"] = False
solver.parameters['gmres']['restart'] = 100

# AMG approximation to 0,0-block inverse
#PETScOptions.set("fieldsplit_u_ksp_type", "richardson")
#PETScOptions.set("fieldsplit_u_ksp_max_it", 1)
##PETScOptions.set("fieldsplit_u_pc_type", "gamg") # Does not work
#PETScOptions.set("fieldsplit_u_pc_type", "ml")
#PETScOptions.set("fieldsplit_u_mg_levels_ksp_type", "chebyshev")
#PETScOptions.set("fieldsplit_u_mg_levels_ksp_max_it", 2)
#PETScOptions.set("fieldsplit_u_mg_levels_pc_type", "sor")
#PETScOptions.set("fieldsplit_u_mg_levels_pc_sor_its", 2)

# LU 0,0-block inverse
PETScOptions.set("fieldsplit_u_ksp_type", "preonly")
PETScOptions.set("fieldsplit_u_pc_type", "lu")
#PETScOptions.set("fieldsplit_u_pc_factor_mat_solver_package", "mumps")

# Compute solution using Ossen approximation
w = Function(W)
solver.set_operator(A)
solver.setup(mp, fp, ap, Lp, bcs_pcd, strategy)
solver.solve(w.vector(), b)
while True:
    u0.assign(w.sub(0, deepcopy=True))
    A, b = assemble_system(a, L, bcs)
    solver.set_operator(A)
    # End when num iterations is zero (residual is small)
    if solver.solve(w.vector(), b) == 0:
        break

# Split the mixed solution using a shallow copy
(u, p) = w.split()

# Save solution in VTK format
#File("velocity.xdmf") << u
#File("pressure.xdmf") << p

# Plot solution
plot(u)
plot(p)
interactive()
