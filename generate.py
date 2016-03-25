from __future__ import print_function

from dolfin import *
import csv
import sys
from scipy.special import jv
from sympy import re, im, sqrt, symbols, lambdify, besselj, sympify

__author__ = 'jh'

note = ''  # to append name of precomputed solution modification
fname = 'tmp.csv'
R = 1.5
# read coeficients from chosen file
infile = open(fname, 'r')
csvreader = csv.reader(infile, delimiter=',')
n_coefs = csvreader.next()[0]
coefs_mult = [sympify(i.replace('*^', 'E')) for i in csvreader.next()]
coefs_bes_mult = [sympify(i.replace('*^', 'E')) for i in csvreader.next()]
coefs_bes = [sympify(i.replace('*^', 'E')) for i in csvreader.next()]
coef_par_max = csvreader.next()[0]
coef_par = csvreader.next()[0]

# for meshName in ['cyl_d1', 'cyl_d2', 'cyl_d3', 'cyl_e3']:
#for meshName in ['cyl_c1', 'cyl_c2', 'cyl_c3']:
for meshName in ['cyl15_3']:
    print('Mesh: '+meshName)
    mesh = Mesh("meshes/" + meshName + ".xml")
    cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
    facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")

    PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)

    f = HDF5File(mpi_comm_world(), 'precomputed/precomputed_'+meshName+note+'.hdf5', 'w')

    # precomputation of Bessel functions=============================================================================
    temp = toc()
    coefs_r_prec = []  # these will be functions in PS
    coefs_i_prec = []  # these will be functions in PS
    c0ex = Expression("(maxc-c*(x[0]*x[0]+x[1]*x[1]))", maxc=Constant(coef_par_max), c=Constant(coef_par))
    f.write(interpolate(c0ex, PS), "parab")
    # plot(interpolate(c0ex, PS), interactive=True, title='Parab')
    for i in range(8):
        r = symbols('r')
        besRe = re(coefs_mult[i] * (coefs_bes_mult[i] * besselj(0, r * coefs_bes[i]) + 1))
        besIm = im(coefs_mult[i] * (coefs_bes_mult[i] * besselj(0, r * coefs_bes[i]) + 1))
        besRe_lambda = lambdify(r, besRe, ['numpy', {'besselj': jv}])
        besIm_lambda = lambdify(r, besIm, ['numpy', {'besselj': jv}])


        class PartialReSolution(Expression):
            def eval(self, value, x):
                rad = float(sqrt(x[0] * x[0] + x[1] * x[1]))
                # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
                value[0] = 0 if near(rad, R) else besRe_lambda(rad)  # do not evaluate on boundaries, it's 0
                # print(value) gives reasonable values


        expr = PartialReSolution()
        print('i = '+str(i)+'   interpolating real part')
        # plot(interpolate(expr, PS), interactive=True, title='Real%d'%i)
        f.write(interpolate(expr, PS), "real%d" % i)


        class PartialImSolution(Expression):
            def eval(self, value, x):
                rad = float(sqrt(x[0] * x[0] + x[1] * x[1]))
                # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
                value[0] = 0 if near(rad, R) else besIm_lambda(rad)  # do not evaluate on boundaries, it's 0


        expr = PartialImSolution()
        print('i = '+str(i)+'   interpolating imaginary part')
        # plot(interpolate(expr, PS), interactive=True, title='Imag%d'%i)
        f.write(interpolate(expr, PS), "imag%d" % i)
        #exit()

    print("Precomputed partial solution functions. Time: %f" % (toc() - temp))


