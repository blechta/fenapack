PCD preconditioner for Navier-Stokes equations
==============================================

This demo is implemented in a single Python file,
:download:`demo_navier-stokes-pcd.py
</../../demo/navier-stokes-pcd/demo_navier-stokes-pcd.py>`,
which contains both the variational forms and the solver.


Underlying mathematics
----------------------

See BRM variant in :ref:`math_background`.


Implementation
--------------

**Only features beyond standard FEniCS usage will be explained
in this document.**

Here comes an artificial boundary condition for PCD operators. This
is the version used for :any:`BRM variant of PCD
<fenapack.preconditioners.PCDPC_BRM>`, i.e. zero Dirichlet value for Laplacian
solve on inlet. Note that it is defined on pressure subspace of the mixed space
``W``.

.. code-block:: python

    # Artificial BC for PCD preconditioner
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)

Then comes standard formulation of the nonlinear equation

.. code-block:: python

    # Arguments and coefficients of the form
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    w = Function(W)
    u_, p_ = split(w)
    n = FacetNormal(mesh)
    nu = Constant(args.viscosity)

    # Nonlinear equation
    F = (
          nu*inner(grad(u_), grad(v))
        + inner(dot(grad(u_), u_), v)
        - p_*div(v)
        - q*div(u_)
    )*dx

We will also mock Newton solver into Picard iteration by passing
Oseen linearization as Jacobian ``J``

.. code-block:: python

    # Picard linearization (one could use full Newton)
    J = (
          nu*inner(grad(u), grad(v))
        + inner(dot(grad(u), u_), v)
        - p*div(v)
        - q*div(u)
    )*dx

"Preconditioner" Jacobian ``J_pc`` features added streamline diffusion
to stabilize algebraic multigrid applied to 00-block

.. code-block:: python

    # Add stabilization for AMG 00-block
    delta = StabilizationParameterSD(w.sub(0), nu)
    J_pc = J + delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx

:math:`L^2` scalar product ("mass matrix") ``mp``, convection operator ``kp``,
and Laplacian ``ap`` to be used by :any:`PCD BRM preconditioner
<fenapack.preconditioners.PCDPC_BRM>` are defined using pressure components
``p``, ``q`` on the mixed space ``W``. They are passed to the class
:py:class:`fenapack.nonlinear_solvers.PCDProblem` which takes care of
assembling the operators on demand.

.. code-block:: python

    # PCD operators
    mp = Constant(1.0/nu)*p*q*dx
    kp = Constant(1.0/nu)*dot(grad(p), u_)*q*dx
    ap = inner(grad(p), grad(q))*dx

    # Collect forms to define nonlinear problem
    problem = PCDProblem(F, [bc0, bc1], J, J_pc, ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)

Now we setup GMRES solver with right-preconditioned Schur complement method
with upper factorization and user preconditioner.

.. code-block:: python

    # Set up linear field split solver
    linear_solver = FieldSplitSolver(W, "gmres")
    linear_solver.parameters["monitor_convergence"] = True
    linear_solver.parameters["relative_tolerance"] = 1e-6
    linear_solver.parameters["nonzero_initial_guess"] = False
    linear_solver.parameters["preconditioner"]["side"] = "right"
    linear_solver.parameters["preconditioner"]["fieldsplit"]["type"] = "schur"
    linear_solver.parameters["preconditioner"]["fieldsplit"]["schur"]["fact_type"] = "upper"
    linear_solver.parameters["preconditioner"]["fieldsplit"]["schur"]["precondition"] = "user"

We fetch :py:class:`petsc4py.PETSc.Options` databases for setting 00- and
11-block subsolvers

.. code-block:: python

    # Set up subsolvers
    OptDB_00, OptDB_11 = linear_solver.get_subopts()

00-block is solver using algebraic multigrid

.. code-block:: python

    OptDB_00["ksp_type"] = "richardson"
    OptDB_00["ksp_max_it"] = 1
    OptDB_00["pc_type"] = "hypre"
    OptDB_00["pc_hypre_type"] = "boomeramg"

PETSc is told to use :py:class:`fenapack.preconditioners.PCDPC_BRM` class
implementing :py:class:`petsc4py.PETSc.PC` interface as a Schur complement
preconditioner

.. code-block:: python

    OptDB_11["ksp_type"] = "preonly"
    OptDB_11["pc_type"] = "python"
    OptDB_11["pc_python_type"] = "fenapack.PCDPC_BRM"

Laplacian solve of PCD is performed using algebraic multigrid

.. code-block:: python

    OptDB_11["PCD_Ap_ksp_type"] = "richardson"
    OptDB_11["PCD_Ap_ksp_max_it"] = 2
    OptDB_11["PCD_Ap_pc_type"] = "hypre"
    OptDB_11["PCD_Ap_pc_hypre_type"] = "boomeramg"

Mass-matrix solve is done using fixed number of Chebyshev iterations with
Jacobi preconditioner. For eigenvalue estimates used for Chebyshev see [1]_,
Lemma 4.3.

.. code-block:: python

    OptDB_11["PCD_Mp_ksp_type"] = "chebyshev"
    OptDB_11["PCD_Mp_ksp_max_it"] = 5
    OptDB_11["PCD_Mp_ksp_chebyshev_eigenvalues"] = "0.5, 2.0"
    OptDB_11["PCD_Mp_pc_type"] = "jacobi"

Finally we invoke Newton solver (although doing Picard iteration)

.. code-block:: python

    # Set up nonlinear solver
    solver = NewtonSolver(linear_solver)
    solver.parameters["relative_tolerance"] = 1e-5

    # Solve problem
    solver.solve(problem, w.vector())

.. [1] Elman H. C., Silvester D. J., Wathen A. J., *Finite Elements and Fast
       Iterative Solvers: With Application in Incompressible Fluid Dynamics*.
       Oxford University Press 2005. 2nd edition 2014.

Complete code
-------------

.. literalinclude:: /../../demo/navier-stokes-pcd/demo_navier-stokes-pcd.py
   :start-after: # Begin demo
