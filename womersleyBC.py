from __future__ import print_function

__author__ = 'jh'
from dolfin import Expression, near, pi
from sympy import I, re, sqrt, exp, symbols, lambdify, besselj
from scipy.special import jv

# file provides some information (mainly averages) about analytic solution for use in womersley_cylinder.py

R = 5.0
r, tm = symbols('r tm')
u = (-43.2592 * r ** 2 +
     (-11.799 + 0.60076 * I) * ((0.000735686 - 0.000528035 * I)
                                * besselj(0, r * (1.84042 + 1.84042 * I)) + 1) * exp(-8 * I * pi * tm) +
     (-11.799 - 0.60076 * I) * ((0.000735686 + 0.000528035 * I)
                                * besselj(0, r * (1.84042 - 1.84042 * I)) + 1) * exp(8 * I * pi * tm) +
     (-26.3758 - 4.65265 * I) * (-(0.000814244 - 0.00277126 * I)
                                 * besselj(0, r * (1.59385 - 1.59385 * I)) + 1) * exp(6 * I * pi * tm) +
     (-26.3758 + 4.65265 * I) * (-(0.000814244 + 0.00277126 * I)
                                 * besselj(0, r * (1.59385 + 1.59385 * I)) + 1) * exp(-6 * I * pi * tm) +
     (-51.6771 + 27.3133 * I) * (-(0.0110653 - 0.00200668 * I)
                                 * besselj(0, r * (1.30138 + 1.30138 * I)) + 1) * exp(-4 * I * pi * tm) +
     (-51.6771 - 27.3133 * I) * (-(0.0110653 + 0.00200668 * I)
                                 * besselj(0, r * (1.30138 - 1.30138 * I)) + 1) * exp(4 * I * pi * tm) +
     (-33.1594 - 95.2423 * I) * ((0.0314408 - 0.0549981 * I)
                                 * besselj(0, r * (0.920212 - 0.920212 * I)) + 1) * exp(2 * I * pi * tm) +
     (-33.1594 + 95.2423 * I) * ((0.0314408 + 0.0549981 * I)
                                 * besselj(0, r * (0.920212 + 0.920212 * I)) + 1) * exp(
          -2 * I * pi * tm) + 1081.48)
# how this works?
u_lambda = lambdify([r, tm], u, ['numpy', {'besselj': jv}])


def average_analytic_velocity(factor):
    return 1081.48 * factor


def average_analytic_velocity_expr(factor):
    return Expression(("0", "0", "factor*(1081.48-43.2592 * (x[0]*x[0]+x[1]*x[1]))"), factor=factor)


def average_analytic_pressure(factor):
    return 641.967*10.0*pi*R*R*factor


class WomersleyProfile(Expression):
    # class no longer used, this inefficient way of generating analytic solution was replaced by
    # precomputation and assembly of Bessel functions (modes)
    def __init__(self, factor):
        self.t = 0
        self.factor = factor

    def eval(self, value, x):
        rad = float(sqrt(x[0] * x[0] + x[1] * x[
            1]))  # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
        value[0] = 0
        value[1] = 0
        value[2] = 0 if near(rad, R) else re(self.factor * u_lambda(rad, self.t))  # do not evaluate on boundaries, it's 0
        # print(x[0], x[1], x[2], rad, value[2])

    def value_shape(self):
        return (3,)

p = -re(641.967 + (15.0987 - 296.542*I)*exp(8*I*pi*tm) + (87.7004 - 497.173*I)*exp(6*I*pi*tm) + (343.229 - 649.393*I)*exp(4*I*pi*tm) + (598.425 - 208.347*I)*exp(2*I*pi*tm) + (598.425 + 208.347*I)*exp(-2*I*pi*tm) + (343.229 + 649.393*I)*exp(-4*I*pi*tm) + (87.7004 + 497.173*I)*exp(-6*I*pi*tm) + (15.0987 + 296.542*I)*exp(-8*I*pi*tm))
p_lambda = lambdify([tm], p)


def average_analytic_pressure_grad(factor):
    return 641.967 * factor


def analytic_pressure_grad(factor, t):
    return factor * p_lambda(t)


def average_analytic_pressure_expr(factor):
    gradient = -641.967 * factor
    return Expression("grad*x[2]", grad=gradient)


def analytic_pressure(factor, t):
    gradient = analytic_pressure_grad(factor, t)
    return Expression("grad*x[2]", grad=gradient)