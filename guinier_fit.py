#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Simple script to perform a Guinier fit on small-angle scattering data

    Fit values:
    I(0) = intensity extrapolated to zero-angle
    Rg = radius of gyration

    Relevant fit range is when q_max * Rg < 1.3
"""

import numpy as np
import matplotlib.pyplot as plt


def read_datafile(f):
    data = np.loadtxt(f, delimiter=',', skiprows=2)
    return data


# Example scattering data on a protein
data = read_datafile('protein_sans.txt')

x = data[:, 0]
y = data[:, 1]

# Set q_min and q_max for fit range
q_min = 0.016
q_max = 0.043

# Parameter to adjust plot range
plot_more = 0.6

cond_data = (x >= q_min * (1 - plot_more)) & (x <= q_max * (1 + plot_more))
x = x[cond_data]
y = y[cond_data]

cond_fit = (x >= q_min) & (x <= q_max)
xFit = x[cond_fit]
yFit = y[cond_fit]

# Linearize data to ln[I(q)] vs. q^2
x = x**2
y = np.log(y)

xFit = xFit**2
yFit = np.log(yFit)

p, cov = np.polyfit(xFit, yFit, 1, cov=True)
yFit = np.polyval(p, xFit)

# Calculate I(0) and Rg values from fit
I0 = np.exp(p[1])
Rg = np.sqrt(-3 * p[0])
qmaxRg = q_max * Rg

# To Do: Add error calculations
# I0_err =
# Rg_err =

# Create Guinier plot with fit
plt.clf()
plt.plot(x, y, 'x')
plt.plot(xFit, yFit)
plt.xlabel(r'${q^2}$ ($\AA^{-2}$)', fontsize=14)
plt.ylabel(r'$ln[I(q)]$', fontsize=14)
plt.title('Guinier Plot', fontsize=14)

results = ('y = a x + b' +
           '\n' + 'a = ' + str(round(p[0], 2)) + ', b = ' + str(round(p[1], 4)) +
           '\n'
           '\n' + 'I(0) = ' + str(round(I0, 4)) +
           '\n' + '$R_g$ = ' + str(round(Rg, 2)) + ' $\AA$' +
           '\n' + '$q_{max}$$R_g$ = ' + str(round(qmaxRg, 2))
           )

plt.annotate(results, xy=(1, 1), xycoords='axes fraction', fontsize=14,
             xytext=(-200, -20), textcoords='offset points',
             ha='left', va='top')

plt.show(block=True)

# Also print out results
print('Fit using y= a x + b')
print('a =', round(p[0], 2), 'b =', round(p[1], 4))
print('q_min^2 =', round(q_min**2, 6), ',', 'q_max^2 =', round(q_max**2, 6))
print('I(0) =', round(I0, 4))
print('Rg =', round(Rg, 2))
print('q_max*Rg =', round(qmaxRg, 2))
