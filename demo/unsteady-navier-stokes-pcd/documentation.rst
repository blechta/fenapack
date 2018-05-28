.. _demo_PCDR_PC_for_unsteady_NS_eqns:

PCD(R) preconditioner for unsteady Navier-Stokes equations
==========================================================

This demo is implemented in two Python files,
:download:`demo_navier-stokes-pcd.py
</../../demo/unsteady-navier-stokes-pcd/demo_unsteady-navier-stokes-pcd.py>`
and
:download:`demo_navier-stokes-pcdr.py
</../../demo/unsteady-navier-stokes-pcd/demo_unsteady-navier-stokes-pcdr.py>`,
so it is easier to make comparison of PCD vs. PCDR.
Each python file contains both the variational forms and the solver.
The differences between the two files are marked by **#PCDR-DIFF** and described
below in detail.


Underlying mathematics
----------------------

See :ref:`math_background`, especially :ref:`math_background_PCDR_extension`.


Implementation
--------------

**This demo extends** :ref:`demo_PCD_PC_for_NS_eqns` **to time-dependent problems.
Only the differences will be explained.**

The nonlinear equation now contains terms from the time derivative
(we use backward implicit Euler).

.. code-block:: python

    # Arguments and coefficients of the form
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    w = Function(W)
    w0 = Function(W)
    u_, p_ = split(w)
    u0_, p0_ = split(w0)
    nu = Constant(args.viscosity)
    idt = Constant(1.0/args.dt)

    # Nonlinear equation
    F = (
          idt*inner(u_ - u0_, v)
        + nu*inner(grad(u_), grad(v))
        + inner(dot(grad(u_), u_), v)
        - p_*div(v)
        - q*div(u_)
    )*dx

The same holds for the Jacobian ``J`` that can be used to mock Newton solver
into Picard iteration.

.. code-block:: python

    # Jacobian
    if args.nls == "picard":
        J = (
              idt*inner(u, v)
            + nu*inner(grad(u), grad(v))
            + inner(dot(grad(u), u_), v)
            - p*div(v)
            - q*div(u)
        )*dx
    elif args.nls == "newton":
        J = derivative(F, w)

If we wish to use any of the :py:class:`PCD BRM preconditioners
<fenapack.preconditioners.PCDPC_BRM>`, then we need to enrich the *convection*
operator ``kp`` by the *reaction* term from the time derivative.

.. code-block:: python

    # PCD operators
    mp = Constant(1.0/nu)*p*q*dx
    kp = Constant(1.0/nu)*(idt*p + dot(grad(p), u_))*q*dx
    ap = inner(grad(p), grad(q))*dx
    if args.pcd_variant == "BRM2":
        n = FacetNormal(mesh)
        ds = Measure("ds", subdomain_data=boundary_markers)
        kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(1)

    # Collect forms to define nonlinear problem
    pcd_assembler = PCDAssembler(J, F, [bc0, bc1],
                                 J_pc, ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)
    problem = PCDNonlinearProblem(pcd_assembler)

**#PCDR-DIFF No. 1:** If we wish to use any of the :py:class:`PCDR BRM preconditioners
<fenapack.preconditioners.PCDRPC_BRM>`, then the *convection* operator ``kp``
remains unchanged, but we need to supply the velocity mass matrix
``mu = idt*inner(u, v)*dx`` to :py:class:`fenapack.assembling.PCDAssembler`.
The pressure gradient ``gp = - p_*div(v)`` does not have to be assembled
as it can be extracted from the Jacobian ``J``.

.. code-block:: python

    # Collect forms to define nonlinear problem
    pcd_assembler = PCDAssembler(J, F, [bc0, bc1],
                                 J_pc, ap=ap, kp=kp, mp=mp, mu=mu, bcs_pcd=bc_pcd)
    assert pcd_assembler.get_pcd_form("gp").phantom # pressure grad obtained from J

**#PCDR-DIFF No. 2:** The fact that we want to use the PCDR preconditioner must be
invoked from options.

.. code-block:: python

    # Set up subsolvers
    PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDRPC_" + args.pcd_variant)

**#PCDR-DIFF No. 3:** The Laplacian solve related to the *reaction* term can be
performed using algebraic multigrid. (The direct solver is used by default if
the following block is not executed.)

.. code-block:: python

    if args.ls == "iterative":
        PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_type", "richardson")
        PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_max_it", 1)
        PETScOptions.set("fieldsplit_p_PCD_Rp_pc_type", "hypre")
        PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_type", "boomeramg")

Try to run

.. code-block:: console

    python3 demo_unsteady-navier-stokes-pcd.py --pcd BRM1
    python3 demo_unsteady-navier-stokes-pcdr.py --pcd BRM1

to see that the results can look like this:

+----------------+-----------------+-----------------+---------------------+----------------+
|  No. of DOF    |      Steps      |   Krylov its    | Krylov its (p.t.s.) |    Time (s)    |
+================+=================+=================+=====================+================+
|    25987       |       25        |      3157       |        126.3        |      99.21     |
+----------------+-----------------+-----------------+---------------------+----------------+
|    25987       |       25        |      1686       |        67.4         |      67.65     |
+----------------+-----------------+-----------------+---------------------+----------------+

Let us remark that the difference in case ``--pcd BRM2`` is not so striking.


Complete code (PCDR version)
----------------------------

.. literalinclude:: /../../demo/unsteady-navier-stokes-pcd/demo_unsteady-navier-stokes-pcdr.py
   :start-after: # Begin demo
