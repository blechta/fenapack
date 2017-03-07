from __future__ import print_function

from dolfin import *
from mshr import *

import os
import sys

path = os.path.join(os.getcwd(), os.path.dirname(__file__))
path = os.path.realpath(path)

resolution = int(sys.argv[1])

end = 150
ratio = 6
inlet = Rectangle(Point(0, -1), Point(2.5, 1))
main  = Rectangle(Point(2.5, -ratio), Point(end, +ratio))
domain = inlet + main

m = generate_mesh(domain, resolution)

print("num vertices:", m.num_vertices())
print("num cells:", m.num_cells())

File(os.path.join(path, "mesh.xml.gz")) << m
