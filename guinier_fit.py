#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Simple program to perform a Guinier fit on small-angle scattering data

Fit values:
I(0) = intensity extrapolated to zero-angle
Rg = radius of gyration

Relevant fit range is when q_max*Rg < 1.3
'''

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def read_datafile():
    '''
    Read in data as comma delimited
    '''
    datafile = input('Datafile name [default=protein_sans.txt]: ') \
        or 'protein_sans.txt'

    data = np.loadtxt(datafile, delimiter=',', skiprows=2)
    xdata = data[:, 0]
    ydata = data[:, 1]
    yerr = data[:, 2]
    return xdata, ydata, yerr


def guinier():
    '''
    Create Guinier plot from data: log(y) vs. x^2
    '''
    # Linearize data to ln[I(q)] vs. q^2
    xdata, ydata, yerr = read_datafile()

    x2 = xdata**2

    # Suppress RuntimeWarning for log(-value)
    with np.errstate(invalid='ignore'):
        logy = np.log(ydata)

    logyerr = yerr / ydata

    return xdata, x2, logy, logyerr


def qrange():
    '''
    Select the fit range: q_min, q_max
    '''
    # Set q_min and q_max for fit range of interest
    q_min = float(input('Input q_min (1/A) [default=0.02]: ') or 0.02)
    q_max = float(input('Input q_max (1/A) [default=0.043]: ') or 0.043)

    return q_min, q_max


def line(x, a, b):
    '''
    Fit function for optimize.curve_fit()
    '''
    return a*x + b


def guinier_fit(xdata, x2, logy, logyerr):
    '''
    Perform Guinier fit to data and return fit values
    '''
    q_min, q_max = qrange()

    cond_fit = (xdata >= q_min) & (xdata <= q_max)
    x2_xlim = x2[cond_fit]
    logy_xlim = logy[cond_fit]
    logyerr_xlim = logyerr[cond_fit]

    popt, pcov = optimize.curve_fit(line, x2_xlim, logy_xlim,
                                    sigma=logyerr_xlim)

    Err_a = pcov[0, 0]**0.5
    Err_b = pcov[1, 1]**0.5

    # Calculate I(0) and Rg values from fit
    I0 = np.exp(popt[1])
    Rg = np.sqrt(-3*popt[0])
    qmaxRg = q_max*Rg
    I0_err = abs(np.exp(popt[1])*Err_b)
    Rg_err = abs(-3/2 * (-3 * popt[0])**(-0.5) * Err_a)

    return x2_xlim, logy_xlim, popt, I0, Rg, qmaxRg, I0_err, Rg_err


def results(popt, I0, Rg, qmaxRg, I0_err, Rg_err):
    '''
    Print fit results from guinier_fit()
    '''
    print('Fit using y= a x + b')
    print('a =', round(popt[0], 2), 'b =', round(popt[1], 4))
    print('I(0) =', round(I0, 4), '+/-', round(I0_err, 4))
    print('Rg =', round(Rg, 2), '+/-', round(Rg_err, 2))
    print('q_max*Rg =', round(qmaxRg, 2))


def plot(x2, logy, logyerr, x2_xlim, logy_xlim,
         popt, I0, Rg, qmaxRg, I0_err, Rg_err):
    '''
    Generate Guinier plot and fit from guinier_fit()
    '''
    # Optional: Adjust the plot zoom-out (0 = no zoom out)
    xpand = 0.2

    plt.clf()
    plt.plot(x2_xlim, line(x2_xlim, popt[0], popt[1]))  # Fit
    plt.errorbar(x2, logy, yerr=logyerr, fmt='k.')  # Data
    plt.title('Guinier Fit')
    plt.xlabel('$q^2$ ($\AA^{-2}$)', fontsize=14)
    plt.ylabel(r'$ln[I(q)]$', fontsize=14)
    plt.xlim(x2_xlim[0] * (1-xpand), x2_xlim[-1] * (1+xpand))
    plt.ylim(logy_xlim[-1] * (1+xpand), logy_xlim[0] * (1-xpand))

    results = ('y = a x + b' + '\n'
               + 'a = ' + str(round(popt[0], 2))
               + ', b = ' + str(round(popt[1], 4))
               + '\n \n'
               + 'I(0) = ' + str(round(I0, 4)) + ' $\pm$ ' + str(round(I0_err, 4))
               + '\n'
               + '$R_g$ = ' + str(round(Rg, 2)) + ' $\pm$ ' + str(round(Rg_err, 2))
               + ' $\AA$'
               + '\n' + '$q_{max}$$R_g$ = ' + str(round(qmaxRg, 2))
               )

    plt.annotate(results, xy=(1, 1), xycoords='axes fraction', fontsize=14,
                 xytext=(-200, -20), textcoords='offset points',
                 ha='left', va='top')

    plt.show(block=True)


# Steps:
xdata, x2, logy, logyerr = guinier()

x2_xlim, logy_xlim, popt, I0, Rg, qmaxRg, I0_err, Rg_err = \
    guinier_fit(xdata, x2, logy, logyerr)

plot(x2, logy, logyerr, x2_xlim, logy_xlim, popt, I0, Rg, qmaxRg,
     I0_err, Rg_err)

results(popt, I0, Rg, qmaxRg, I0_err, Rg_err)
