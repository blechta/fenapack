.. _demo_PCD_PC_for_NS_eqns:

PCD preconditioner for Navier-Stokes equations
==============================================

This demo is implemented in a single Python file,
:download:`demo_navier-stokes-pcd.py
</../../demo/navier-stokes-pcd/demo_navier-stokes-pcd.py>`,
which contains both the variational forms and the solver.


Underlying mathematics
----------------------

See :ref:`math_background`.


Implementation
--------------

**Only features beyond standard FEniCS usage will be explained
in this document.**

Here comes an artificial boundary condition for PCD operators. Zero Dirichlet
condition for Laplacian solve is applied either on inlet or outlet, depending
on the variant of PCD. Note that it is defined on pressure subspace of the
mixed space ``W``.

.. code-block:: python

    # Artificial BC for PCD preconditioner
    if args.pcd_variant == "BRM1":
        bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)
    elif args.pcd_variant == "BRM2":
        bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 2)

Then comes standard formulation of the nonlinear equation

.. code-block:: python

    # Arguments and coefficients of the form
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    w = Function(W)
    u_, p_ = split(w)
    nu = Constant(args.viscosity)

    # Nonlinear equation
    F = (
          nu*inner(grad(u_), grad(v))
        + inner(dot(grad(u_), u_), v)
        - p_*div(v)
        - q*div(u_)
    )*dx

We will provide a possibility to mock Newton solver into Picard iteration by
passing Oseen linearization as Jacobian ``J``

.. code-block:: python

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

"Preconditioner" Jacobian ``J_pc`` features added streamline diffusion
to stabilize 00-block if algebraic multigrid is used. Otherwise we can
pass ``None`` as a precoditioner Jacobian to use the system matrix for
preparing the preconditioner.

.. code-block:: python

    # Add stabilization for AMG 00-block
    if args.ls == "iterative":
        delta = StabilizationParameterSD(w.sub(0), nu)
        J_pc = J + delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx
    elif args.ls == "direct":
        J_pc = None

:math:`L^2` scalar product ("mass matrix") ``mp``, convection operator ``kp``,
and Laplacian ``ap`` to be used by :any:`PCD BRM preconditioner
<fenapack.preconditioners.PCDPC_BRM>` are defined using pressure components
``p``, ``q`` on the mixed space ``W``. They are passed to the class
:py:class:`fenapack.assembling.PCDAssembler` which takes care of
assembling the operators on demand.

.. code-block:: python

    # PCD operators
    mp = Constant(1.0/nu)*p*q*dx
    kp = Constant(1.0/nu)*dot(grad(p), u_)*q*dx
    ap = inner(grad(p), grad(q))*dx
    if args.pcd_variant == "BRM2":
        n = FacetNormal(mesh)
        ds = Measure("ds", subdomain_data=boundary_markers)
        kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(1)

    # Collect forms to define nonlinear problem
    pcd_assembler = PCDAssembler(J, F, [bc0, bc1],
                                 J_pc, ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)
    problem = PCDNonlinearProblem(pcd_assembler)

Now we create GMRES preconditioned with PCD, set the tolerance, enable
monitoring of residual during Krylov iterarations, and set the maximal
dimension of Krylov subspaces.

.. code-block:: python

    # Set up linear solver (GMRES with right preconditioning using Schur fact)
    linear_solver = PCDKrylovSolver(comm=mesh.mpi_comm())
    linear_solver.parameters["relative_tolerance"] = 1e-6
    PETScOptions.set("ksp_monitor")
    PETScOptions.set("ksp_gmres_restart", 150)

Next we choose a variant of PCD according to a parameter value

.. code-block:: python

    # Set up subsolvers
    PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDPC_" + args.pcd_variant)

00-block solve and PCD Laplacian solve can be performed using algebraic
multigrid

.. code-block:: python

    if args.ls == "iterative":
        PETScOptions.set("fieldsplit_u_ksp_type", "richardson")
        PETScOptions.set("fieldsplit_u_ksp_max_it", 1)
        PETScOptions.set("fieldsplit_u_pc_type", "hypre")
        PETScOptions.set("fieldsplit_u_pc_hypre_type", "boomeramg")
        PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_type", "richardson")
        PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_max_it", 2)
        PETScOptions.set("fieldsplit_p_PCD_Ap_pc_type", "hypre")
        PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")

PCD mass matrix solve can be efficiently performed using Chebyshev iteration
preconditioned by Jacobi method. The eigenvalue estimates come from [1]_,
Lemma 4.3. **Don't forget to change them appropriately when changing
dimension/element. Neglecting this can lead to substantially worse
convergence rates.**

.. code-block:: python

        PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
        PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_max_it", 5)
        PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.0")
        PETScOptions.set("fieldsplit_p_PCD_Mp_pc_type", "jacobi")

The direct solver is used by default if the aforementioned blocks
are not executed. FENaPack tries to pick MUMPS by default and following
parameter enables very verbose output.

.. code-block:: python

    elif args.ls == "direct" and args.mumps_debug:
        # Debugging MUMPS
        PETScOptions.set("fieldsplit_u_mat_mumps_icntl_4", 2)
        PETScOptions.set("fieldsplit_p_PCD_Ap_mat_mumps_icntl_4", 2)
        PETScOptions.set("fieldsplit_p_PCD_Mp_mat_mumps_icntl_4", 2)

Let the linear solver use the options

.. code-block:: python

    # Apply options
    linear_solver.set_from_options()

Finally we invoke a Newton solver modification suitable to be used used
with PCD solver.

.. code-block:: python

    # Set up nonlinear solver
    solver = PCDNewtonSolver(linear_solver)
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
