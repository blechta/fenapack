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

from dolfin import *
from field_split import FieldSplitSolver

import sys

try:
    num_refinements = int(sys.argv[1])
except IndexError:
    num_refinements = 0

# Load mesh and subdomains
mesh = Mesh("../data/dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "../data/dolfin_fine_subdomains.xml.gz")

# Refine
# NOTE: Only works sequentially
sub_domains_old = []
for i in range(num_refinements):
    mesh = adapt(mesh, CellFunction('bool', mesh, True))
    # BUG in DOLFIN: this is needed to prevent memory corruption
    sub_domains_old.append(sub_domains)
    sub_domains = adapt(sub_domains, mesh)

# Define function spaces
P1 = FunctionSpace(mesh, "Lagrange", 1)
B  = FunctionSpace(mesh, "Bubble", 3)
gdim = mesh.geometry().dim()
V  = MixedFunctionSpace(gdim*(P1+B,))
Q  = FunctionSpace(mesh, "CG",  1)
Mini = MixedFunctionSpace((V, Q))
info('Dimension of the function space %g' % Mini.dim())

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(Mini.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(Mini.sub(0), inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
zero = Constant(0.0)
bc2 = DirichletBC(Mini.sub(1), zero, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = TrialFunctions(Mini)
(v, q) = TestFunctions(Mini)
#nu = Constant(1e-1)
nu = Expression('x[1] < x1 ? nu1 : nu2', x1=0.5, nu1=1e1, nu2=2e-1)
f  = Constant((0.0, 0.0))
a  = (nu*inner(grad(u), grad(v)) + div(v)*p + q*div(u))*dx
L  = inner(f, v)*dx

# Add Oseen-like convection
u0 = Function(V)
a += inner(dot(grad(u), u0), v)*dx

# Operators for PCD preconditioner
Mp = p*q*dx
Fp = (nu*inner(grad(p), grad(q)) + dot(grad(p), u0)*q)*dx
Ap = inner(grad(p), grad(q))*dx
Lp = Constant(0.0)*q*dx # Dummy right-hand side

# Assemble
A, b = assemble_system(a, L, bcs)

# Setup fieldsplit solver
solver = FieldSplitSolver(Mini)
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
PETScOptions.set("fieldsplit_u_pc_factor_mat_solver_package", "mumps")

# Compute solution using Ossen approximation
w = Function(Mini)
solver.set_operator(A)
solver.setup(Mp, Fp, Ap, Lp, [bc2])
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
File("velocity.xdmf") << u
File("pressure.xdmf") << p

# Plot solution
plot(u)
plot(p)
interactive()
