# -*- coding: utf-8 -*-
from defcon import *
from dolfin import *
from petsc4py import PETSc
from matplotlib import pyplot

from fenapack import PCDKSP
from fenapack import PCDProblem
from fenapack import StabilizationParameterSD


# Parameters
inner_solvers = "direct"
#inner_solvers = "iterative"
#pcd_variant = "BRM1"
pcd_variant = "BRM2"
#linearization = "picard"
linearization = "newton"


class NavierStokesProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = Mesh(comm, "../../data/sudden_expansion.xml.gz")
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", triangle, 2)
        Qe = FiniteElement("CG", triangle, 1)
        Ze = MixedElement([Ve, Qe])
        Z  = FunctionSpace(mesh, Ze)
        return Z

    def parameters(self):
        Re = Constant(0)
        return [(Re, "Re", r"$\mathrm{Re}$")]

    def residual(self, z, params, w):
        (u, p) = split(z)
        (v, q) = split(w)

        Re = params[0]
        mesh = z.function_space().mesh()

        # Variational form
        F = (
              1.0/Re * inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - p*div(v)*dx
            - q*div(u)*dx
            )

        # Trial functions
        u_, p_ = TrialFunctions(z.function_space())

        # Fetch problem BCs and facet markers
        bcs = self.boundary_conditions(z.function_space(), self.parameters())
        colours = self._colours

        # Artificial BC for PCD preconditioner
        if pcd_variant == "BRM1":
            bc_pcd = DirichletBC(z.function_space().sub(1), 0.0, colours, 1)
        elif pcd_variant == "BRM2":
            bc_pcd = DirichletBC(z.function_space().sub(1), 0.0, colours, 2)

        # Jacobian
        if linearization == "picard":
            J = (
                  1.0/Re * inner(grad(u_), grad(v))*dx
                + inner(grad(u_)*u, v)*dx
                - p_*div(v)*dx
                - q*div(u_)*dx
            )
        elif linearization == "newton":
            J = derivative(F, z)

        if inner_solvers == "iterative":
            # Add stabilization for AMG 00-block
            iRe = Expression("1./Re", Re=Re, degree=0, mpi_comm=mesh.mpi_comm())
            delta = StabilizationParameterSD(z.sub(0), iRe)
            J_pc = J + delta*inner(grad(u_)*u, grad(v)*u)*dx
        else:
            J_pc = None

        # PCD operators
        mp = Re*p_*q*dx
        kp = Re*dot(grad(p_), u)*q*dx
        ap = inner(grad(p_), grad(q))*dx
        if pcd_variant == "BRM2":
            n = FacetNormal(mesh)
            ds = Measure("ds", subdomain_data=colours)
            kp -= Re*dot(u, n)*p_*q*ds(1)
            #kp -= Re*dot(u, n)*p_*q*ds(0)  # TODO: Is this beneficial?

        # Store what needed for later
        self._J = J
        self._pcd_problem = PCDProblem(F, bcs, J, J_pc,
                                       ap=ap, kp=kp, mp=mp,
                                       bcs_pcd=bc_pcd)

        return F

    def boundary_conditions(self, Z, params):
        comm = Z.mesh().mpi_comm()

        # Facet markers
        colours = FacetFunction("size_t", Z.mesh())
        walls = CompiledSubDomain("on_boundary", mpi_comm=comm)
        inflow = CompiledSubDomain("on_boundary && near(x[0], 0.0)", mpi_comm=comm)
        outflow = CompiledSubDomain("on_boundary && near(x[0], 150.0)", mpi_comm=comm)
        colours.set_all(999)      # interior facets
        walls.mark(colours, 0)    # no-slip facets
        inflow.mark(colours, 1)   # inlet facets
        outflow.mark(colours, 2)  # outlet facets

        # BCs
        poiseuille = Expression(("-(x[1] + 1) * (x[1] - 1)", "0.0"), degree=2, mpi_comm=comm)
        bc_inflow = DirichletBC(Z.sub(0), poiseuille, colours, 1)
        bc_wall = DirichletBC(Z.sub(0), (0, 0), colours, 0)

        bcs = [bc_inflow, bc_wall]

        # Store markers for lates use
        self._colours = colours

        return bcs

    def functionals(self):
        def sqL2(z, params):
            (u, p) = split(z)
            # FIXME: Why is here |z|^2
            j = assemble(inner(z, z)*dx)
            return j

        return [(sqL2, "sqL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return Function(Z)

    def number_solutions(self, params):
        return float("inf")

        #Re = params[0]
        #if   Re < 18:  return 1
        #elif Re < 41:  return 3
        #elif Re < 75:  return 5
        #elif Re < 100: return 8
        #else:          return float("inf")

    def solver_parameters(self, params, klass):
        opts = {
            "snes_max_it": 50,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "ksp_converged_reason": None,
            "ksp_monitor": None,
            "ksp_gmres_restart": 64,
            "ksp_rtol": 1.0e-5,
            "ksp_atol": 0.0,
        }

        if inner_solvers == "iterative":
            opts.update({
                "fieldsplit_u_ksp_type": "richardson",
                "fieldsplit_u_ksp_max_it": 1,
                "fieldsplit_u_pc_type": "hypre",
                "fieldsplit_u_pc_hypre_type": "boomeramg",
                "fieldsplit_p_pc_python_type": "fenapack.PCDPC_" + pcd_variant,
                "fieldsplit_p_PCD_Ap_ksp_type": "richardson",
                "fieldsplit_p_PCD_Ap_ksp_max_it": 2,
                "fieldsplit_p_PCD_Ap_pc_type": "hypre",
                "fieldsplit_p_PCD_Ap_pc_hypre_type": "boomeramg",
                "fieldsplit_p_PCD_Mp_ksp_type": "chebyshev",
                "fieldsplit_p_PCD_Mp_ksp_max_it": 5,
                "fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues": "0.5, 2.0",
                "fieldsplit_p_PCD_Mp_pc_type": "jacobi",
            })

        return opts

    def jacobian(self, F, state, params, test, trial):
        return self._J

    def solver(self, problem, solver_params, prefix="", **kwargs):
        # Create linear solver
        Z = problem.u.function_space()
        ksp = PCDKSP(Z)

        # Create nonlinear solver
        solver = SNUFLSolver(problem, prefix=prefix,
                             solver_parameters=solver_params,
                             **kwargs)

        # Switch ksp (reuse operators)
        oldksp = solver.snes.ksp
        ksp.setOperators(*oldksp.getOperators())
        ksp.setOptionsPrefix(oldksp.getOptionsPrefix())
        solver.snes.ksp = ksp
        solver.snes.setFromOptions()

        # Initilize PCD (only possible with assembled operators)
        _ = as_backend_type(Function(Z).vector()).vec()  # FIXME: Remove this hack!
        solver.jacobian(solver.snes, _, *solver.snes.ksp.getOperators())
        ksp.init_pcd(self._pcd_problem)

        return solver


if __name__ == "__main__":
    # Debugging options
    #set_log_level(INFO)
    set_log_level(PROGRESS)
    #set_log_level(TRACE)
    SubSystemsManager.init_petsc()
    PETSc.Sys.pushErrorHandler("traceback")
    PETScOptions.set("options_left")

    dc = DeflatedContinuation(problem=NavierStokesProblem(), teamsize=1, verbose=True)
    dc.run(values={"Re": [1.0]})
    #dc.run(values={"Re": linspace(10.0, 100.0, 181)})

    # FIXME: This is not what we want possibly; an average over WORLD,
    #        thus biased by master thread
    list_timings(TimingClear_keep, [TimingType_wall, TimingType_user])

    #dc.bifurcation_diagram("sqL2")
    #pyplt.title(r"Bifurcation diagram for sudden expansion in a channel")
    #pyplt.savefig("bifurcation.pdf")
