# Set up DOLFIN's default global parameters
from dolfin import parameters
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True

# Import FENaPack modules
from fenapack.field_split import *
from fenapack.preconditioners import *
from fenapack.nonlinear_solvers import *
from fenapack.stabilization import *
