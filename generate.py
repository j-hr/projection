from __future__ import print_function
from dolfin import *
import sys
from sympy import I, re, im, sqrt, symbols, lambdify, besselj
from scipy.special import jv
__author__ = 'jh'

meshName = sys.argv[1]
factor = 1.0
print('Mesh: '+meshName)
mesh = Mesh("meshes/" + meshName + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")

R = 5.0

PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)

f = HDF5File(mpi_comm_world(), 'precomputed_'+meshName+'.hdf5', 'w')

# precomputation of Bessel functions=============================================================================
temp = toc()
coefs_mult = [(-11.799 + 0.60076 * I), (-11.799 - 0.60076 * I), (-26.3758 - 4.65265 * I),
              (-26.3758 + 4.65265 * I), (-51.6771 + 27.3133 * I), (-51.6771 - 27.3133 * I),
              (-33.1594 - 95.2423 * I), (-33.1594 + 95.2423 * I)]
coefs_bes_mult = [(0.000735686 - 0.000528035 * I), (0.000735686 + 0.000528035 * I),
                  -(0.000814244 - 0.00277126 * I), -(0.000814244 + 0.00277126 * I),
                  -(0.0110653 - 0.00200668 * I), -(0.0110653 + 0.00200668 * I), (0.0314408 - 0.0549981 * I),
                  (0.0314408 + 0.0549981 * I)]
coefs_bes = [(1.84042 + 1.84042 * I), (1.84042 - 1.84042 * I), (1.59385 - 1.59385 * I), (1.59385 + 1.59385 * I),
             (1.30138 + 1.30138 * I), (1.30138 - 1.30138 * I), (0.920212 - 0.920212 * I),
             (0.920212 + 0.920212 * I)]
# coefs_exp = [-8, 8, 6, -6, -4, 4, 2, -2]
coefs_r_prec = []  # these will be functions in PS
coefs_i_prec = []  # these will be functions in PS
c0ex = Expression("factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))", factor=factor)
f.write(interpolate(c0ex, PS), "parab")
for i in range(8):
    r = symbols('r')
    besRe = re(factor * coefs_mult[i] * (coefs_bes_mult[i] * besselj(0, r * coefs_bes[i]) + 1))
    besIm = im(factor * coefs_mult[i] * (coefs_bes_mult[i] * besselj(0, r * coefs_bes[i]) + 1))
    besRe_lambda = lambdify(r, besRe, ['numpy', {'besselj': jv}])
    besIm_lambda = lambdify(r, besIm, ['numpy', {'besselj': jv}])


    class PartialReSolution(Expression):
        def eval(self, value, x):
            rad = float(sqrt(x[0] * x[0] + x[1] * x[
                1]))  # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
            value[0] = 0 if near(rad, R) else besRe_lambda(rad)  # do not evaluate on boundaries, it's 0
            # print(value) gives reasonable values


    expr = PartialReSolution()
    print('i = '+str(i)+'   interpolating real part')
    f.write(interpolate(expr, PS), "real%d" % i)


    class PartialImSolution(Expression):
        def eval(self, value, x):
            rad = float(sqrt(x[0] * x[0] + x[1] * x[
                1]))  # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
            value[0] = 0 if near(rad, R) else besIm_lambda(rad)  # do not evaluate on boundaries, it's 0


    expr = PartialImSolution()
    print('i = '+str(i)+'   interpolating imaginary part')
    f.write(interpolate(expr, PS), "imag%d" % i)

print("Precomputed partial solution functions. Time: %f" % (toc() - temp))


