# CONFIGURATION
from void_analysis import plot, tools, snapedit, catalogue, simulation_tools
from void_analysis.catalogue import *
#from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.real_clusters import getClusterSkyPositions
from void_analysis import massconstraintsplot
from void_analysis.simulation_tools import (
    ngPerLBin,
    SnapshotGroup,
    redshift_space_positions,
    get_los_pos_for_snapshot,
    get_los_positions_for_all_catalogues,
)
from void_analysis.plot import draw_ellipse, plot_los_void_stack
from void_analysis import context
from void_analysis.tools import ratio_where_finite, product_where_finite
from matplotlib import transforms
import matplotlib.ticker
from matplotlib.ticker import NullFormatter
from matplotlib import cm
from matplotlib import patches
import matplotlib.lines as mlines
import matplotlib.colors as colors
import pickle
import numpy as np
try:
    import seaborn as sns
    seabornColormap = sns.color_palette("colorblind",as_cmap=True)
except:
    sns = None
    seabornColormap = [
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
        '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9'
    ]

import pynbody
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy
import os
import sys
try:
    import emcee
except:
    emcee = None
    print("WARNING: emcee not found. Some functionality is disabled.")

from fractions import Fraction

from void_analysis.cosmology import Hz, D1, D1_CPT, f_lcdm, Omega_z, Ez2
from fractions import Fraction
import astropy

#-------------------------------------------------------------------------------
# COSMOLOGY FUNCTIONS


def get_D_coefficients(Om,order=1,n2 = -1/143,n3a = -4/275,
                       n3b = -269/17875,n4a = 0.0,n4b=0.0,n4c = 0.0,n4d = 0.0,
                       n5a=0.0,n5b=0.0,n5c=0.0,n5d=0.0,n5e=0.0,n5f=0.0,
                       n5g=0.0,n5h=0.0,n5i=0.0,return_all = False,**kwargs):
    """
    Compute the coefficients of the time-dependent parts of the LPT solutions
    of different orders. Coefficients are computed by solving the 
    Einstein-de-Sitter (EdS) universe case which is analytic. Other solutions
    are typically proportional to this with some power of Omega
    
    Parameters:
        Om (float): Matter density parameter at the relevant redshift.
        order (int): Order of Lagrangian Perturbation Theory
        nNx (float): Exponent of Om(z)^nNx for order N, solution x (alphabetic).
        reuturn_all (bool): If true, return all coefficient up to the specified
                            order. Otherwise, return only coefficients of the
                            specified order.
    
    Returns:
        List of floats: giving the solutions DNx(t) for each spatial solution
                        in LPT.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_D_coefficients
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    if order >= 1:
        D10 = D1(0,Om,**kwargs)
        if order == 1:
            return [D10]
    if order >= 2:
        D20 = -(3/7)*(Om**n2)*D10**2
        if order == 2:
            if return_all:
                return [D10, D20]
            else:
                return [D20]
    if order >=3:
        D3a0 = -(1/3)*(Om**n3a)*D10**3
        D3b0 = (10/21)*(Om**n3b)*D10**3
        if order == 3:
            if return_all:
                return [D10, D20, D3a0, D3b0]
            else:
                return [D3a0, D3b0]
    if order >= 4:
        Gfactor = Fraction(2,3)
        power1 = Fraction(2,3)
        deriv_factor = lambda p: p**2 + p/3 - Fraction(2,3)
        d_factor4 = deriv_factor(4*power1)
        C10 = Fraction(1,1)
        C20 = Fraction(-3,7)
        C3a0 = Fraction(-1,3)
        C3b0 = Fraction(10,21)
        # Work out exact fractional pre-factors, using EdS special case:
        C4a0 = 2*Gfactor*C10*(2*C10**3 - C3a0)/d_factor4
        C4b0 = 2*Gfactor*C10*(2*C10*D20 - 2*C10**3 - C3b0)/d_factor4
        C4c0 = Gfactor*(2*C10**2*C20 - C20**2)/d_factor4
        C4d0 = Gfactor*(C10**4 - 2*C10**2*C20)/d_factor4
        # Get coefficients:
        D4a0 = C4a0*D10**4*(Om**n4a)
        D4b0 = C4b0*D10**4*(Om**n4b)
        D4c0 = C4c0*D10**4*(Om**n4c)
        D4d0 = C4d0*D10**4*(Om**n4d)
        if order == 4:
            if return_all:
                return [D10, D20, D3a0, D3b0, D4a0, D4b0, D4c0, D4d0]
            else:
                return [D4a0, D4b0, D4c0, D4d0]
    if order >= 5:
        d_factor5 = deriv_factor(5*power1)
        # Work out exact fractional pre-factors, using EdS special case:
        C5a0 = -2*Gfactor*C10*(4*C10**4 - 2*C10*C3a0 + C4a0)/d_factor5
        C5b0 = -2*Gfactor*C10*(4*C10**2*C20 - 4*C10**4 - 2*C10*C3b0
                                + C4b0)/d_factor5
        C5c0 = -2*Gfactor*C10*(2*C10**2*C20 - C20**2 + C4c0)/d_factor5
        C5d0 = -2*Gfactor*C10*(C10**4 - 2*C10**2*C20 + C4d0)/d_factor5
        C5e0 = 2*Gfactor*(2*C10**3*C20 - C3a0*C20 + C3a0*C10**2)/d_factor5
        C5f0 = 2*Gfactor*(2*C10*C20**2 - 2*C10**3*C20 - C3b0*C20 
                          + C3b0*C10**2)/d_factor5
        C5g0 = 2*Gfactor*(C10**5 - C10**2*C3a0)/d_factor5
        C5h0 = 2*Gfactor*( - C10**5 + C10**3*C20 - C10**2*C3b0)/d_factor5
        C5i0 = 2*Gfactor*(C10**3*C20 - C10*C20**2)/d_factor5
        # Get coefficients:
        D5a0 = C5a0*D10**5*(Om**n5a)
        D5b0 = C5b0*D10**5*(Om**n5b)
        D5c0 = C5c0*D10**5*(Om**n5c)
        D5d0 = C5d0*D10**5*(Om**n5d)
        D5e0 = C5e0*D10**5*(Om**n5e)
        D5f0 = C5f0*D10**5*(Om**n5f)
        D5g0 = C5g0*D10**5*(Om**n5g)
        D5h0 = C5h0*D10**5*(Om**n5h)
        D5i0 = C5i0*D10**5*(Om**n5i)
        if order == 5:
            if return_all:
                return [D10, D20, D3a0, D3b0, D4a0, D4b0, D4c0, D4d0, D5a0, 
                        D5b0, D5c0, D5d0, D5e0, D5f0, D5g0, D5h0, D5i0]
            else:
                return [D5a0, D5b0, D5c0, D5d0, D5e0, D5f0, D5g0, D5h0, D5i0]

def get_ic_polynomial_coefficients(order,Om=0.3,n2 = -1/143,n3a = -4/275,
                          n3b = -269/17875,**kwargs):
    """
    Computes the coefficients of the initial conditions polynomial for LPT, 
    under the assumption that we use a perturbative estimate for the 
    final density.
    
    Parameters:
        order (int): Order of LPT
        Om (float): Matter density parameter
        nNx (float): Exponent of Om(z)^nNx for order N, solution x (alphabetic).
    
    Returns:
        A1...AN (floats): Coefficients of the initial conditions polynomial
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_ic_polynomial_coefficients
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    coeffs = get_D_coefficients(
        Om,order=order,n2 = n2,n3a = n3a,n3b = n3b,return_all = True,**kwargs
    )
    if order >= 1:
        D10 = coeffs[0]
        A1 = -3
        if order == 1:
            return A1
    if order >= 2:
        D20 = coeffs[1]
        A2 = 3*(D10 - D20/D10)
        if order == 2:
            return A1, A2
    if order >=3:
        [D3a0, D3b0] = coeffs[2:4]
        A3 = 3*D20 - D10**2 - D3a0/D10 - 3*D3b0/D10
        if order == 3:
            return A1, A2, A3
    if order >=4:
        [D4a0, D4b0, D4c0, D4d0] = coeffs[4:8]
        A4 = -(D4a0 + 3*D4b0 + 3*D4c0 + 3*D4d0)/D10
        if order == 4:
            return A1, A2, A3, A4
    if order >= 5:
        [D5a0, D5b0, D5c0, D5d0, D5e0, D5f0, D5g0, D5h0, D5i0] = coeffs[8:17]
        A5 = (-D20*(D3a0 + 3*D3b0) - D10*(D4a0 + 3*D4b0 + 3*D4c0 + 3*D4d0)
           - (D5a0 + 3*D5b0 + 3*D5c0 + 3*D5d0 + D5e0 + 3*D5f0
              + D5g0 + 3*D5h0 + 3*D5i0))/D10
        if order == 5:
            return A1, A2, A3, A4, A5

def get_nonperturbative_polynomial_coefficients(
        order,Om=0.3,n2 = -1/143,n3a = -4/275,n3b = -269/17875,**kwargs):
    """
    Computed initial conditions polynomial coefficients, using the
    non-perturbative expression for the final density in the spherically
    symmetric case.
    
    Parameters:
        order (int): Order of LPT
        Om (float): Matter density parameter
        nNx (float): Exponent of Om(z)^nNx for order N, solution x (alphabetic).
    
    Returns:
        B1...BN (floats): Coefficients of the initial conditions polynomial
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_nonperturbative_polynomial_coefficients
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    coeffs = get_D_coefficients(
        Om,order=order,n2 = n2,n3a = n3a,n3b = n3b,return_all = True,**kwargs
    )
    if order >= 1:
        D10 = coeffs[0]
        B1 = D10
        if order == 1:
            return B1
    if order >= 2:
        D20 = coeffs[1]
        B2 = D20
        if order == 2:
            return B1, B2
    if order >=3:
        [D3a0, D3b0] = coeffs[2:4]
        B3 = D3a0/3 + D3b0
        if order == 3:
            return B1, B2, B3
    if order >=4:
        [D4a0, D4b0, D4c0, D4d0] = coeffs[4:8]
        B4 = (D4a0/3 + D4b0 + D4c0 + D4d0)
        if order == 4:
            return B1, B2, B3, B4
    if order >= 5:
        [D5a0, D5b0, D5c0, D5d0, D5e0, D5f0, D5g0, D5h0, D5i0] = coeffs[8:17]
        B5 = (D5a0/3 + D5b0 + D5c0 + D5d0 + D5e0/3 + D5f0
               + D5g0/3 + D5h0 + D5i0)
        if order == 5:
            return B1, B2, B3, B4, B5

def get_taylor_polynomial_coefficients(
        order,Om=0.3,n2 = -1/143,n3a = -4/275,n3b = -269/17875,**kwargs):
    """
    Computed initial conditions polynomial coefficients, using a Taylor-expanded
    expression for the density field. This turned out not to work so well
    because the Taylor expansion converges very slowly, so
    should probably be avoided, but we retain it here for future use.
    
    Parameters:
        order (int): Order of LPT
        Om (float): Matter density parameter
        nNx (float): Exponent of Om(z)^nNx for order N, solution x (alphabetic).
    
    Returns:
        B1...BN (floats): Coefficients of the initial conditions polynomial
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_taylor_polynomial_coefficients
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    coeffs = get_D_coefficients(
        Om,order=order,n2 = n2,n3a = n3a,n3b = n3b,return_all = True,**kwargs
    )
    if order >= 1:
        D10 = coeffs[0]
        B1 = -3*D10
        if order == 1:
            return B1
    if order >= 2:
        D20 = coeffs[1]
        B2 = 6*D10**2 - 3*D20
        if order == 2:
            return B1, B2
    if order >=3:
        [D3a0, D3b0] = coeffs[2:4]
        B3 = -10*D10**3 + 12*D10*D20 - D3a0 - 3*D3b0
        if order == 3:
            return B1, B2, B3
    if order >=4:
        [D4a0, D4b0, D4c0, D4d0] = coeffs[4:8]
        B4 = (15*D10**4 - 30*D10**2*D20 + 4*D10*D3a0 + 12*D10*D3b0 + 6*D20**2
               - D4a0 - 3*D4b0 - 3*D4c0 - 3*D4d0)
        if order == 4:
            return B1, B2, B3, B4
    if order >= 5:
        [D5a0, D5b0, D5c0, D5d0, D5e0, D5f0, D5g0, D5h0, D5i0] = coeffs[8:17]
        B5 = (-21*D10**5 + 60*D10**3*D20 - 10*D10**2*D3a0 - 30*D10**2*D3b0 
               - 30*D10*D20**2 + 4*D10*D4a0 + 12*D10*D4b0 + 12*D10*D4c0
               + 12*D10*D4d0 + 4*D20*D3a0 + 12*D20*D3b0 - D5a0 - 3*D5b0
               - 3*D5c0 - 3*D5d0 - D5e0 - 3*D5f0 - D5g0 - 3*D5h0 - 3*D5i0)
        if order == 5:
            return B1, B2, B3, B4, B5

def find_suitable_solver_bounds(f,RHS,D10,taylor_expand=True,iter_max = 10,
                                uupp = None):
    """
    Finds bounds for the solution of f = RHS. Used when solving the initial
    conditions polynomial, so that we can guarantee finding a solution 
    with brents method.
    
    Parameters:
        f (function): Function we wish to solve
        RHS (float or array): Values for which we wish to solve f = RHS.
        D10 (float): Value of the D1 function.
        taylor_expand (bool): If True, using a taylor-expanded version of the 
                              density for f. Otherwise, uses the 
                              non-perturbative expression.
        iter_max(int): Maximum number of iterations before we give up and throw
                       an error. Exceeding this probably means no solution 
                       exists, which typically only happens if the polynomial
                       was incorrectly computed.
        uupp (float or None): Existing upper bound. If None, will attemtp to
                              find one. Otherwise, this will be used and the
                              function simply finds a lower bound only.
    
    Returns:
        ulow (float), uupp (float): Lower and upper bounds for the solution.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Unit test: test_find_suitable_solver_bounds_valid
    """
    RHS_max = np.max(RHS)
    if taylor_expand:
        if uupp is None:
            uupp = 1.0
        if RHS_max <= 0:
            ulow = 0.0
        else:
            ulow = -RHS_max/(3*D10)
            count = 0
            while f(ulow) < RHS_max:
                ulow *= 10
                count += 1
                if count > iter_max:
                    raise Exception("Failing to find valid boundary for " + 
                                    "solver.")
    else:
        ulow = -1.0
        if uupp is None:
            uupp = RHS_max/D10
            count = 0
            while f(uupp) < RHS_max:
                uupp *= 10
                count += 1
                if count > iter_max:
                    raise Exception(
                        "Failing to find valid boundary for solver."
                    )
    return ulow, uupp

def get_initial_condition_non_perturbative(Delta,order=1,Om=0.3,n2 = -1/143,
                                           n3a = -4/275,n3b = -269/17875,
                                           use_linear_on_fail=False,
                                           taylor_expand = True,**kwargs):
    """
    Solve the equation for the LPT initial conditions. Using a non-perturbative
    expression for Psi_r/q that avoids having to expand and approximate r/q.
    Effectively resums this series to avoid convergence issues.
    
    Parameters:
        Delta (float or array): Densities at which to solve for the initial
                                condition.
        order (int): Perturbation order to find the solution for
        Om (float): Matter density parameter
        n2 (float): Exponent for the second roder growth function
        n3a (float): Exponent for the first 3rd-order growth function.
        n3b (float): Exponent for the second 3rd order growth function.
        use_linear_on_fail (bool): If True, defaults to the linear solution
                                   when we fail to find a solution for 2LPT.
                                   Otherwise, generates warnings and produces
                                   nan values as the solution.
        taylor_expand (bool): If True (default), uses a Taylor-expanded form
                              of the relationship between density and 
                              Psi_r/q, avoiding inconsistency problems.
    Returns:
        float or array: Solution for S_1r/r at this value of Delta
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_initial_condition_non_perturbative
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    if taylor_expand:
        RHS = Delta # We re-arranged so that the RHS is Delta_f in the 
        # Taylor expanded version, as this turns out to be easier.
    else:
        RHS = 1.0/np.cbrt(1.0 + Delta) - 1.0
    D10 = D1(0,Om,**kwargs)
    if order == 1:
        # Linear solution, known analytically:
        if taylor_expand:
            return -RHS/(3*D10)
        else:
            return RHS/D10
    if order == 2:
        # Quadratic solution. This is known analytically, but has the potential
        # to be non-existant. In which case, we shouldn't return an imaginary
        # number, but instead report this to the user, since it indicates
        # that we have gone beyond the bounds where 2LPT is applicable:
        if taylor_expand:
            B1, B2 = get_taylor_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        else:
            B1, B2 = get_nonperturbative_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        B0 = -RHS
        discriminant = B1**2 - 4*B2*B0
        # Split between vector and scalar cases:
        if np.isscalar(Delta):
            if discriminant < 0:
                if use_linear_on_fail:
                    if taylor_expand:
                        return -RHS/(3*D10)
                    else:
                        return RHS/D10
                else:
                    print("2LPT solution for initial conditions " + 
                          "does not exist.")
                    return np.nan
            else:
                # Two solutions exist. In general, only the negative solution
                # will actually be consistent with 3LPT, so we should pick that 
                # one:
                return (-B1 - np.sqrt(discriminant))/(2*B2)
        else:
            have_no_solution = (discriminant < 0)
            solution = np.zeros(Delta.shape)
            if use_linear_on_fail:
                if taylor_expand:
                    solution[have_no_solution] = -RHS[have_no_solution]/(3*D10)
                else:
                    solution[have_no_solution] = RHS[have_no_solution]/D10
            else:
                if np.any(have_no_solution):
                    print("Warning: no 2LPT solution for some values of " + 
                          "input. Returning nan for these values.")
                solution[have_no_solution] = np.nan
            have_solution = np.logical_not(have_no_solution)
            solution[have_solution] = (
                -B1 - np.sqrt(discriminant[have_solution])
            )/(2*B2)
            return solution
    if order == 3:
        # Cubic solution. For even vaguely sensible parameters, 
        # there is always a unique solution. We can use the linear solution as 
        # an initial guess and then solve numerically:
        # Compute coefficient of the polynomial:
        if taylor_expand:
            B1, B2, B3 = get_taylor_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        else:
            B1, B2, B3 = get_nonperturbative_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        B0 = -RHS
        f = lambda u: B3*u**3 + B2*u**2 + B1*u
        # Figure out the correct bounds for the problem:
        ulow, uupp = find_suitable_solver_bounds(
            f,RHS,D10,taylor_expand=taylor_expand
        )
        # Solve numerically:
        if np.isscalar(Delta):
            solution = scipy.optimize.brentq(lambda u: f(u) + B0,ulow,uupp)
        else:
            solution = np.array(
                [scipy.optimize.brentq(lambda u: f(u) + C,ulow,uupp)
                for C in B0]
            )
        return solution
    if order == 4:
        B0 = -RHS
        if taylor_expand:
            B1, B2, B3, B4 = get_taylor_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        else:
            B1, B2, B3, B4 = get_nonperturbative_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        f = lambda u: B4*u**4 + B3*u**3 + B2*u**2 + B1*u
        # Check we aren't below the minimum possible density for 4LPT:
        p4p = lambda u: 4*B4*u**3 + 3*B3*u**2 + 2*B2*u + B1
        turning_point = scipy.optimize.fsolve(p4p,0.5)[0]
        fturn = f(turning_point)
        if taylor_expand:
            # Polynomial gives Delta_f directly:
            delta_min4 = fturn
        else:
            # Polynomial gives Psi_r/q at turning point, so compute Delta_f:
            delta_min4 = 1.0/(1.0 + fturn)**3 - 1.0
        # Figure out the correct bounds for the problem, 
        # overriding the automatic choice with the turning point of the 
        # polynomial, since we don't want to go beyond that:
        ulow, uupp = find_suitable_solver_bounds(
            f,RHS,D10,taylor_expand=taylor_expand,uupp = turning_point
        )
        # Split between scalar and array cases:
        if np.isscalar(Delta):
            if Delta < delta_min4:
                if use_linear_on_fail:
                    return RHS/D10
                else:
                    print("4LPT solution for initial conditions " + 
                          "does not exist.")
                    return np.nan
            else:
                return scipy.optimize.brentq(lambda u: f(u) + B0,ulow,uupp)
        else:
            have_no_solution = (Delta < delta_min4)
            solution = np.zeros(Delta.shape)
            if use_linear_on_fail:
                solution[have_no_solution] = RHS[have_no_solution]/D10
            else:
                if np.any(have_no_solution):
                    print("Warning: no 4LPT solution for some values of " + 
                          "input. Returning nan for these values.")
                solution[have_no_solution] = np.nan
            have_solution = np.logical_not(have_no_solution) & (Delta != 0.0)
            # Solution:
            solution[have_solution] = np.array(
                [scipy.optimize.brentq(lambda u: f(u) + C,ulow,uupp)
                for C in B0[have_solution]]
            )
            return solution
    if order == 5:
        B0 = -RHS
        if taylor_expand:
            B1, B2, B3, B4, B5 = get_taylor_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        else:
            B1, B2, B3, B4, B5 = get_nonperturbative_polynomial_coefficients(
                order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
            )
        f = lambda u: B5*u**5 + B4*u**4 + B3*u**3 + B2*u**2 + B1*u
        # Figure out the correct bounds for the problem:
        ulow, uupp = find_suitable_solver_bounds(
            f,RHS,D10,taylor_expand=taylor_expand
        )
        # Solve:
        if np.isscalar(Delta):
            solution = scipy.optimize.brentq(lambda u: f(u) + B0,ulow,uupp)
        else:
            solution = np.array(
                [scipy.optimize.brentq(lambda u: f(u) + C,ulow,uupp)
                for C in B0]
            )
        return solution

def get_initial_condition(Delta,order=1,Om=0.3,n2 = -1/143,n3a = -4/275,
                          n3b = -269/17875,use_linear_on_fail=False,
                          **kwargs):
    """
    Solve the equation for the LPT initial conditions. At first and 2nd order
    this can be done analytically, but at 3rd order a cubic must be solved, 
    which we do numerically.
    
    Parameters:
        Delta (float or array): Densities at which to solve for the initial
                                condition.
        order (int): Perturbation order to find the solution for
        Om (float): Matter density parameter
        n2 (float): Exponent for the second roder growth function
        n3a (float): Exponent for the first 3rd-order growth function.
        n3b (float): Exponent for the second 3rd order growth function.
        use_linear_on_fail (bool): If True, defaults to the linear solution
                                   when we fail to find a solution for 2LPT.
                                   Otherwise, generates warnings and produces
                                   nan values as the solution.
    Returns:
        float or array: Solution for S_1r/r at this value of Delta
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_initial_condition
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    if order == 1:
        # Linear solution, known analytically:
        D10 = D1(0,Om,**kwargs)
        return -Delta/(3*D10)
    if order == 2:
        # Quadratic solution. This is known analytically, but has the potential
        # to be non-existant. In which case, we shouldn't return an imaginary
        # number, but instead report this to the user, since it indicates
        # that we have gone beyond the bounds where 2LPT is applicable:
        [D10, D20] = get_D_coefficients(Om,order=order,n2 = n2,n3a = n3a,
                                        n3b = n3b,return_all = True,**kwargs)
        A1, A2 = get_ic_polynomial_coefficients(
            order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
        )
        A0 = -Delta/D10
        discriminant = A1**2 - 4*A2*A0
        # Split between vector and scalar cases:
        if np.isscalar(Delta):
            if discriminant < 0:
                if use_linear_on_fail:
                    return -Delta/(3*D10)
                else:
                    print("2LPT solution for initial conditions " + 
                          "does not exist.")
                    return np.nan
            else:
                # Two solutions exist. In general, only the negative solution
                # will actually be consistent with 3LPT, so we should pick that 
                # one:
                return (-A1 - np.sqrt(discriminant))/(2*A2)
        else:
            have_no_solution = (discriminant < 0)
            solution = np.zeros(Delta.shape)
            if use_linear_on_fail:
                solution[have_no_solution] = -Delta[have_no_solution]/(3*D10)
            else:
                if np.any(have_no_solution):
                    print("Warning: no 2LPT solution for some values of " + 
                          "input. Returning nan for these values.")
                solution[have_no_solution] = np.nan
            have_solution = np.logical_not(have_no_solution)
            solution[have_solution] = (
                -A1 - np.sqrt(discriminant[have_solution])
            )/(2*A2)
            return solution
    if order == 3:
        # Cubic solution. For even vaguely sensible parameters, 
        # there is always a unique solution. We can use the linear solution as 
        # an initial guess and then solve numerically:
        # Compute coefficient of the polynomial:
        [D10, D20, D3a0, D3b0] = get_D_coefficients(Om,order=order,n2 = n2,
                                                    n3a = n3a,n3b = n3b,
                                                    return_all = True,**kwargs)
        A1, A2, A3 = get_ic_polynomial_coefficients(
            order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
        )
        A0 = -Delta/D10
        # Solve numerically:
        guess = -Delta/(3*D10)
        f = lambda u: A3*u**3 + A2*u**2 + A1*u
        if np.isscalar(Delta):
            solution = scipy.optimize.fsolve(lambda u: f(u) + A0,guess)[0]
        else:
            solution = np.array(
                [scipy.optimize.fsolve(lambda u: f(u) + C,x0)[0]
                for C, x0 in zip(A0,guess)]
            )
        return solution
    if order == 4:
        [D10, D20, D3a0, D3b0, D4a0, D4b0, D4c0, D4d0] = get_D_coefficients(
            Om,order=order,n2 = n2,n3a = n3a,n3b = n3b,return_all = True,
            **kwargs
            )
        A0 = -Delta/D10
        A1, A2, A3, A4 = get_ic_polynomial_coefficients(
            order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
        )
        f = lambda u: A4*u**4 + A3*u**3 + A2*u**2 + A1*u
        # Check we aren't below the minimum possible density for 4LPT:
        p4p = lambda u: 4*A4*u**3 + 3*A3*u**2 + 2*A2*u + A1
        turning_point = scipy.optimize.fsolve(p4p,0.5)[0]
        delta_min4 = f(turning_point)
        # Figure out the correct bounds for the problem:
        Delta_max = np.max(Delta)
        ulow = -Delta_max/(3*D10)
        count = 0
        while f(ulow) < Delta_max/D10:
            ulow *= 10
            count += 1
            if count > 10:
                raise Exception("Failing to find valid boundary for solver.")
        uupp = turning_point # To prevent getting the wrong solution branch
        # Split between scalar and array cases:
        if np.isscalar(Delta):
            if Delta < delta_min4:
                if use_linear_on_fail:
                    return -Delta/(3*D10)
                else:
                    print("4LPT solution for initial conditions " + 
                          "does not exist.")
                    return np.nan
            else:
                return scipy.optimize.brentq(lambda u: f(u) + A0,ulow,uupp)
        else:
            have_no_solution = (Delta < delta_min4)
            solution = np.zeros(Delta.shape)
            if use_linear_on_fail:
                solution[have_no_solution] = -Delta[have_no_solution]/(3*D10)
            else:
                if np.any(have_no_solution):
                    print("Warning: no 4LPT solution for some values of " + 
                          "input. Returning nan for these values.")
                solution[have_no_solution] = np.nan
            have_solution = np.logical_not(have_no_solution)
            solution[have_solution] = np.array(
                [scipy.optimize.brentq(lambda u: f(u) + C,ulow,uupp)
                for C in A0[have_solution]]
            )
            return solution
    if order == 5:
        [D10, D20, D3a0, D3b0, D4a0, D4b0, D4c0, D4d0, D5a0, 
         D5b0, D5c0, D5d0, D5e0, D5f0, D5g0, D5h0, D5i0] = get_D_coefficients(
            Om,order=order,n2 = n2,n3a = n3a,n3b = n3b,return_all = True,
            **kwargs
            )
        A0 = -Delta/D10
        A1, A2, A3, A4, A5 = get_ic_polynomial_coefficients(
            order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
        )
        f = lambda u: A5*u**5 + A4*u**4 + A3*u**3 + A2*u**2 + A1*u
        # Solve numerically:
        guess = -Delta/(3*D10)
        # Figure out the correct bounds for the problem:
        Delta_max = np.max(Delta)
        ulow = -Delta_max/(3*D10)
        uupp = 1
        count = 0
        while f(ulow) < Delta_max/D10:
            ulow *= 10
            count += 1
            if count > 10:
                raise Exception("Failing to find valid boundary for solver.")
        # Solve:
        if np.isscalar(Delta):
            solution = scipy.optimize.brentq(lambda u: f(u) + A0,ulow,uupp)
        else:
            solution = np.array(
                [scipy.optimize.brentq(lambda u: f(u) + C,ulow,uupp)
                for C in A0]
            )
        return solution

def get_S1r(Delta_r,rval,Om,order=1,n2 = -1/143,n3a = -4/275,n3b = -269/17875,
            perturbative_ics = False,taylor_expand=True,
            force_linear_ics = False,**kwargs):
    """
    Compute the spatial part of the first order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of 
                                  radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
        n2 (float): Exponent for the second roder growth function
        n3a (float): Exponent for the first 3rd-order growth function.
        n3b (float): Exponent for the second 3rd order growth function.
        perturbative_ics (bool): If True, attempt to correct for the q/r
                              ratio with perturbative corrections to the
                              solutions at each order.
        taylor_expand (bool): If True (default), use a Taylor expanded
                              approximation of the relationship between Delta_f
                              and Psi_r/q, to avoid inconsistency issues
                              with the order of LPT expansion. Otherwise, use
                              the non-perturbative relationship.
    Returns:
        float or array: Value of S_{1r}
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_S1r
    """
    D10 = D1(0,Om,**kwargs)
    # Solve for the initial conditions, possibly numerically:
    if perturbative_ics:
        S1r = rval*get_initial_condition(
            Delta_r,order=order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,**kwargs
        )
    else:
        if force_linear_ics:
            S1r = rval*get_initial_condition_non_perturbative(
                Delta_r,order=1,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,
                taylor_expand=taylor_expand,**kwargs
            )
        else:
            S1r = rval*get_initial_condition_non_perturbative(
                Delta_r,order=order,Om=Om,n2 = n2,n3a = n3a,n3b = n3b,
                taylor_expand=taylor_expand,**kwargs
            )
    return S1r

def get_S2r(Delta_r,rval,Om,n2 = -1/143,n3a = -4/275,n3b = -269/17875,order=2,
            perturbative_ics = False,S1r=None,taylor_expand=True,
            **kwargs):
    """
    Compute the spatial part of the second order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of 
                                  radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
        n2 (float): Exponent for the second roder growth function
        n3a (float): Exponent for the first 3rd-order growth function.
        n3b (float): Exponent for the second 3rd order growth function.
        perturbative_ics (bool): If True, attempt to correct for the q/r
                              ratio with perturbative corrections to the
                              solutions at each order.
        taylor_expand (bool): If True (default), use a Taylor expanded
                              approximation of the relationship between Delta_f
                              and Psi_r/q, to avoid inconsistency issues
                              with the order of LPT expansion. Otherwise, use
                              the non-perturbative relationship.
    Returns:
        float or array: Value of S_{2r}
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_S2r
    """
    # Solve for initial conditions (numerically if 3rd order):
    if S1r is None:
        S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                      n3b = n3b,perturbative_ics = perturbative_ics,
                      taylor_expand=taylor_expand,**kwargs)
    S2r = ratio_where_finite(S1r**2,rval,undefined_value=0.0)
    if perturbative_ics:
        if order >= 3:
            S2r = S2r + D10*ratio_where_finite(S1r**3,rval**2,
                                               undefined_value=0.0)
        if order >= 4:
            D20 = get_D_coefficients(Om,order=2,return_all = False,**kwargs)[0]
            S2r += (D20 + D10**2)*ratio_where_finite(S1r**4,rval**3,
                                               undefined_value=0.0)
        if order >= 5:
            [D3a0, D3b0] = get_D_coefficients(Om,order=3,return_all = False,
                                              **kwargs)
            S2r += (3*D10*D20 + D3a0/3 + D3b0 + D10**3)*ratio_where_finite(
                S1r**5,rval**4,undefined_value=0.0
            )
    return S2r

def get_S3r(Delta_r,rval,Om,order=3,perturbative_ics = False,
            n2 = -1/143,n3a = -4/275,n3b = -269/17875,
            S1r = None,taylor_expand=True,**kwargs):
    """
    Compute the spatial part of the third order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of 
                                  radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
        S1r (float or array): Precomputed value of S1r (otherwise, this is
                              computed from scratch)
        perturbative_ics (bool): If True, attempt to correct for the q/r
                              ratio with perturbative corrections to the
                              solutions at each order.
        taylor_expand (bool): If True (default), use a Taylor expanded
                              approximation of the relationship between Delta_f
                              and Psi_r/q, to avoid inconsistency issues
                              with the order of LPT expansion. Otherwise, use
                              the non-perturbative relationship.
                                               
    Returns:
        S3ar (float or array): Value of S_{3ar}
        S3br (float or array): Value of S_{3br}
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_S3r
    """
    D10 = D1(0,Om,**kwargs)
    if S1r is None:
        S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                      n3b = n3b,perturbative_ics = False,
                      taylor_expand=taylor_expand,**kwargs)
    S3ar = ratio_where_finite(S1r**3,3*rval**2,undefined_value=0.0)
    S3br = ratio_where_finite(S1r**3,rval**2,undefined_value=0.0)
    if perturbative_ics:
        if order >= 4:
            S3ar += 2*D10*ratio_where_finite(S1r**4,3*rval**3)
            S3br += 2*D10*ratio_where_finite(S1r**4,rval**3)
        if order >= 5:
            D20 = get_D_coefficients(Om,order=2,return_all = False,**kwargs)[0]
            S3ar += (2*D20 + 3*D10**2)*ratio_where_finite(S1r**5,3*rval**4)
            S3br += (2*D20 + 3*D10**2)*ratio_where_finite(S1r**5,rval**4)
    return S3ar, S3br

def get_S4r(Delta_r,rval,Om,order=4,perturbative_ics = False,taylor_expand=True,
            S1r = None,n2 = -1/143,n3a = -4/275,n3b = -269/17875,**kwargs):
    """
    Compute the spatial part of the fourth order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of 
                                  radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
        S1r (float or array): Precomputed value of S1r (otherwise, this is
                              computed from scratch)
        perturbative_ics (bool): If True, attempt to correct for the q/r
                              ratio with perturbative corrections to the
                              solutions at each order.
        taylor_expand (bool): If True (default), use a Taylor expanded
                              approximation of the relationship between Delta_f
                              and Psi_r/q, to avoid inconsistency issues
                              with the order of LPT expansion. Otherwise, use
                              the non-perturbative relationship.
    Returns:
        S4ar (float or array): Value of S_{4ar}
        S4br (float or array): Value of S_{4br}
        S4cr (float or array): Value of S_{4cr}
        S4dr (float or array): Value of S_{4dr}
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_S4r
    """
    D10 = D1(0,Om,**kwargs)
    if S1r is None:
        S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                      n3b = n3b,perturbative_ics = False,
                      taylor_expand=taylor_expand,**kwargs)
    S4ar = ratio_where_finite(S1r**4,3*rval**3,undefined_value=0.0)
    S4br = ratio_where_finite(S1r**4,rval**3,undefined_value=0.0)
    S4cr = ratio_where_finite(S1r**4,rval**3,undefined_value=0.0)
    S4dr = ratio_where_finite(S1r**4,rval**3,undefined_value=0.0)
    if perturbative_ics:
        if order >= 5:
            S4ar += D10*ratio_where_finite(S1r**5,rval**4)
            S4br += 3*D10*ratio_where_finite(S1r**5,rval**4)
            S4cr += 3*D10*ratio_where_finite(S1r**5,rval**4)
            S4dr += 3*D10*ratio_where_finite(S1r**5,rval**4)
    return S4ar, S4br, S4cr, S4dr

def get_S5r(Delta_r,rval,Om,order=5,perturbative_ics = False,
            taylor_expand=True,S1r = None,
            n2 = -1/143,n3a = -4/275,n3b = -269/17875,**kwargs):
    """
    Compute the spatial part of the fifth order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Note: currently no higher order corrections, so changing order doesn't
    alter anything. However, we retain it in case we want to extend to higher
    orders, which would entail futher corrections.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of 
                                  radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
        S1r (float or array): Precomputed value of S1r (otherwise, this is
                              computed from scratch)
        perturbative_ics (bool): If True, attempt to correct for the q/r
                              ratio with perturbative corrections to the
                              solutions at each order.
        taylor_expand (bool): If True (default), use a Taylor expanded
                              approximation of the relationship between Delta_f
                              and Psi_r/q, to avoid inconsistency issues
                              with the order of LPT expansion. Otherwise, use
                              the non-perturbative relationship.
    Returns:
        S54ar (float or array): Value of S_{5ar}
        S5br (float or array): Value of S_{5br}
        S5cr (float or array): Value of S_{5cr}
        S5dr (float or array): Value of S_{5dr}
        S5er (float or array): Value of S_{5er}
        S5fr (float or array): Value of S_{5fr}
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_S5r
    """
    D10 = D1(0,Om,**kwargs)
    if S1r is None:
        S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                      n3b = n3b,perturbative_ics = False,
                      taylor_expand=taylor_expand,**kwargs)
    S5ar = ratio_where_finite(S1r**5,3*rval**4,undefined_value=0.0)
    S5br = ratio_where_finite(S1r**5,rval**4,undefined_value=0.0)
    S5cr = ratio_where_finite(S1r**5,rval**4,undefined_value=0.0)
    S5dr = ratio_where_finite(S1r**5,rval**4,undefined_value=0.0)
    S5er = ratio_where_finite(S1r**5,3*rval**4,undefined_value=0.0)
    S5fr = ratio_where_finite(S1r**5,rval**4,undefined_value=0.0)
    S5gr = ratio_where_finite(S1r**5,3*rval**4,undefined_value=0.0)
    S5hr = ratio_where_finite(S1r**5,rval**4,undefined_value=0.0)
    S5ir = ratio_where_finite(S1r**5,rval**4,undefined_value=0.0)
    return S5ar, S5br, S5cr, S5dr, S5er, S5fr, S5gr, S5hr, S5ir

def get_psi_n_r(Delta_r,rval,n,z=0,Om=0.3,order=None,n2 = -1/143,
                n3a = -4/275,n3b = -269/17875,S1r=None,return_all=False,
                **kwargs):
    """
    Compute the nth order correction to the displacement field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of 
                                  radius
        rval (float or array): Radius at which to compute, must match Delta_r
        n (int): Order of the correction to compute.
        z (float): Redshift at which to compute the displacement field.
        Om (float): Value of Omega_matter
        order (int): Order of perturbation theory being applied. If not 
                     supplied, assumed to be equal to n. This can matter, if we
                     enable correct_ics = True, since the perturbations change
                     with each order.
        n2 (float): Exponent of Omega_m(z) used to approximate D_2(t)
        n3a (float): Exponent of Omega_m(z) used to approximate D_{3a}(t)
        n3b (float): Exponent of Omega_m(z) used to approximate D_{3b}(t)
        kwargs: Keyword arguments passed to get_S1r, get_S2r, get_S3r functions.

    Return:
        float or array: Correction to the displacement field of order n
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_psi_n_r
    """
    order_list = []
    if order is None:
        order = n
    if order not in [1,2,3,4,5] or n not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    if n > order:
        raise Exception("n <= order is required.")
    if n >= 1:
        D1_val = D1(z,Om,**kwargs)
        if S1r is None:
            S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                          n3b = n3b,**kwargs)
        Psi_q1 = D1_val*S1r
        order_list.append(Psi_q1)
        if n == 1:
            if return_all:
                return order_list
            else:
                return Psi_q1
    if n >= 2:
        Omz = Omega_z(z,Om,**kwargs)
        D2_val = get_D_coefficients(Om,order=2,return_all = False,**kwargs)[0]
        S2r = get_S2r(Delta_r,rval,Om,order=order,n2=n2,n3a = n3a,
                      n3b = n3b,S1r=S1r,**kwargs)
        Psi_q2 = D2_val*S2r
        order_list.append(Psi_q2)
        if n == 2:
            if return_all:
                return order_list
            else:
                return Psi_q2
    if n >=3:
        D3a_val, D3b_val = get_D_coefficients(
            Om,order=3,return_all = False,**kwargs
        )
        S3ar, S3br = get_S3r(Delta_r,rval,Om,order=order,S1r=S1r,**kwargs)
        Psi_q3 = D3a_val*S3ar + D3b_val*S3br
        order_list.append(Psi_q3)
        if n == 3:
            if return_all:
                return order_list
            else:
                return Psi_q3
    if n >=4:
        all_D4 = get_D_coefficients(
            Om,order=4,return_all = False,**kwargs
        )
        all_S4 = get_S4r(
            Delta_r,rval,Om,order=order,S1r=S1r,**kwargs
        )
        Psi_q4 = np.sum([D*S for D, S in zip(all_D4, all_S4)],0)
        order_list.append(Psi_q4)
        if n == 4:
            if return_all:
                return order_list
            else:
                return Psi_q4
    if n >=5:
        all_D5 = get_D_coefficients(
            Om,order=5,return_all = False,**kwargs
        )
        all_S5 = get_S5r(
            Delta_r,rval,Om,order=order,S1r=S1r,**kwargs
        )
        Psi_q5 = np.sum([D*S for D, S in zip(all_D5, all_S5)],0)
        order_list.append(Psi_q5)
        if n == 5:
            if return_all:
                return order_list
            else:
                return Psi_q5

def get_delta_lpt(Delta_r,z=0,Om=0.3,order=1,return_all=False,**kwargs):
    """
    Compute the final density field estimate in perturbation theory, given
    the input final density field. This is mainly used to aid in verifying the
    accuracy of the result
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of radius
        z (float): Redshift at which to compute the displacement field.
        Om (float): Value of Omega_matter
        order (int): Order of expansion for LPT. Only 1,2, or 3 implemented
        return_all (bool): Default False (return only the density estimate). If
                           True, return the corrections to the density
                           with for each order of pertubation theory
                           separately
    Returns:
        float or array (return_all = False): Final density field estimate
        3 floats or arrays (return_all = True): Perturbative corrections to the
                                                density field at each order
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_delta_lpt
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    # Ratio of Psi_r/r:
    if not return_all:
        Psi_r_rat = spherical_lpt_displacement(1.0,Delta_r,order=order,
                                               fixed_delta=True,Om=Om,**kwargs)
        # Exact spherical result for the density field
        #return (-3*Psi_r_rat + 3*Psi_r_rat**2 - Psi_r_rat**3)
        return 1.0/(1.0 + Psi_r_rat)**3 - 1.0
    else:
        # Displacement field corrections/r at each order:
        S1r = get_S1r(Delta_r,1.0,Om,order=order,**kwargs) # Precompute 
        # First order:
        Psi_r1 = get_psi_n_r(Delta_r,1.0,1,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        Delta_1 = -3*Psi_r1
        if order == 1:
            return Delta_1
        # Second Order:
        Psi_r2 = get_psi_n_r(Delta_r,1.0,2,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        Delta_2 = -3*Psi_r2 + 3*Psi_r1**2
        if order == 2:
            return Delta_1, Delta_2
        # Third order:
        Psi_r3 = get_psi_n_r(Delta_r,1.0,3,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        Psi_r2_un = get_psi_n_r(Delta_r,1.0,2,z=z,Om=Om,order=2,S1r=S1r,
                             **kwargs)
        if order > 3:
            # Use the full, corrected solution for lower orders:
            Delta_3 = -3*Psi_r3 + 6*Psi_r1*Psi_r2 - Psi_r1**3
        else:
            # Use only the solution without higher order corrections:
            Delta_3 = -3*Psi_r3 + 6*Psi_r1*Psi_r2_un - Psi_r1**3
        if order == 3:
            return Delta_1, Delta_2, Delta_3
        # Fourth order:
        Psi_r4 = get_psi_n_r(Delta_r,1.0,4,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        Psi_r2_un = get_psi_n_r(Delta_r,1.0,2,z=z,Om=Om,order=2,S1r=S1r,
                             **kwargs)
        Psi_r3_un = get_psi_n_r(Delta_r,1.0,3,z=z,Om=Om,order=3,S1r=S1r,
                             **kwargs)
        if order > 4:
            Delta_4 = (-3*Psi_r4 + 6*Psi_r3*Psi_r1 + 3*(Psi_r2**2) 
                       - 3*Psi_r1**2*Psi_r2)
        else:
            Delta_4 = (-3*Psi_r4 + 6*Psi_r3_un*Psi_r1 + 3*(Psi_r2_un**2) 
                       - 3*Psi_r1**2*Psi_r2_un)
        if order == 4:
            return Delta_1, Delta_2, Delta_3, Delta_4
        # Fifth order:
        Psi_r5 = get_psi_n_r(Delta_r,1.0,5,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        Psi_r2_un = get_psi_n_r(Delta_r,1.0,2,z=z,Om=Om,order=2,S1r=S1r,
                             **kwargs)
        Psi_r3_un = get_psi_n_r(Delta_r,1.0,3,z=z,Om=Om,order=3,S1r=S1r,
                             **kwargs)
        Psi_r4_un = get_psi_n_r(Delta_r,1.0,3,z=z,Om=Om,order=4,S1r=S1r,
                             **kwargs)
        Delta_5 = (-3*Psi_r5 + 6*Psi_r1*Psi_r4_un + 6*Psi_r2_un*Psi_r3_un
                   -3*Psi_r1*Psi_r2_un**2 - 3*Psi_r1**2*Psi_r3_un)
        return Delta_1, Delta_2, Delta_3, Delta_4, Delta_5

def get_eulerian_ratio_from_lagrangian_ratio(quant_list_q,Psi_q_list,order,
                                             expand_denom_only=False):
    """
    Computes the ratio Q/r where Q is some quantity determined from LPT (usually
    displacement, Psi, or velocity, v), and r is the Eulerian radius, assuming
    spherical symmetry and that Q is the same ratio in Lagrangian co-ordinates
    (ie, with respect to q, the Lagrangian radius).
    
    This function is used by process_radius to control whether we output the 
    results in Lagrangian or Eulerian co-ordinates.
    
    Parameters:
        quant_list_q (list of floats/arrays): List containing the quantity in 
                                              Lagrangian co-ordinates at each 
                                              order
        Psi_q_list (list of floats/arrays): List containing the displacement
                                            field corrections at each order.
        order (int): LPT order. Should match the lengths of the lists above.
        expand_denom_only (bool): If True, only the denominator is expanded
                                  (useful if the numerator represents an exactly
                                   known quantity). Otherwise, we expand both
                                   numerator and denominator.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_get_eulerian_ratio_from_lagrangian_ratio
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    if order >= 1:
        v_r_q1 = quant_list_q[0]
        if expand_denom_only:
            qor = 1
            v_r_q = v_r_q1
            v_r_r = qor*v_r_q
        else:
            v_r_r = np.zeros(v_r_q1.shape)
            v_r_r += v_r_q1
        if order == 1:
            return v_r_r
    if order >=2:
        Psi_r_q1 = Psi_q_list[0]
        v_r_q2 = quant_list_q[1]
        if expand_denom_only:
            qor += - Psi_r_q1
            v_r_q += v_r_q2
            v_r_r = qor*v_r_q
        else:
            v_r_r += -Psi_r_q1*v_r_q1 + v_r_q2
        if order == 2:
            return v_r_r
    if order >=3:
        Psi_r_q2 = Psi_q_list[1]
        v_r_q3 = quant_list_q[2]
        if expand_denom_only:
            qor += Psi_r_q1**2 - Psi_r_q2
            v_r_q += v_r_q3
            v_r_r = qor*v_r_q
        else:
            v_r_r += (Psi_r_q1**2*v_r_q1 - Psi_r_q1*v_r_q2 - Psi_r_q2*v_r_q1
                         + v_r_q3)
        if order == 3:
            return v_r_r
    if order >=4:
        Psi_r_q3 = Psi_q_list[2]
        v_r_q4 = quant_list_q[3]
        if expand_denom_only:
            qor += 2*Psi_r_q1*Psi_r_q2 - Psi_r_q3 - Psi_r_q1**3
            v_r_q += v_r_q4
            v_r_r = qor*v_r_q
        else:
            v_r_r += (-Psi_r_q1**3*v_r_q1 + Psi_r_q1**2*v_r_q2
                        + 2*Psi_r_q1*Psi_r_q2*v_r_q1 - Psi_r_q1*v_r_q3
                        - Psi_r_q2*v_r_q2 - Psi_r_q3*v_r_q1 + v_r_q4)
        if order == 4:
            return v_r_r
    if order >= 5:
        Psi_r_q4 = Psi_q_list[3]
        v_r_q5 = quant_list_q[4]
        if expand_denom_only:
            qor += (Psi_r_q1**4 - 3*Psi_r_q1**2*Psi_r_q2 + Psi_r_q2**2 
                    + 2*Psi_r_q1*Psi_r_q3 - Psi_r_q4)
            v_r_q += v_r_q5
            v_r_r = qor*v_r_q
        else:
            v_r_r += (Psi_r_q1**4*v_r_q1 - Psi_r_q1**3*v_r_q2
                      - 3*Psi_r_q1**2*Psi_r_q2*v_r_q1 + Psi_r_q1**2*v_r_q3
                      + 2*Psi_r_q1*Psi_r_q2*v_r_q2 + 2*Psi_r_q1*Psi_r_q3*v_r_q1
                      - Psi_r_q1*v_r_q4 + Psi_r_q2**2*v_r_q1 - Psi_r_q2*v_r_q3
                      - Psi_r_q3*v_r_q2 - Psi_r_q4*v_r_q1 + v_r_q5)
    return v_r_r

def process_radius(r,Psi_q,quant_q = None,radial_fraction=True,
                   eulerian_radius=True,taylor_expand=True,
                   order=None,expand_denom_only=False):
    """Return the Value of Psi_r, either as a fraction of the radius
    or on it's own, in the correct co-ordinates (Lagrangian vs Eulerian)
    
    Parameters:
        r (float or array): Radius at which to evaluate Psi_r
        Psi_q (float or array): Displacement field as a fraction of Lagangian 
                                radial co-ordinate, q
        quant_q (float, array, or None): Quantity as a fraction of
                               Lagrangian co-ordinate that we want to return.
                               If not supplied, we assume that this quantity
                               if just Psi_q
        radial_fracion (bool): If true, return the dimensionless radial
                               fraction Psi_r/r otherwise return Psi_r alone
                               which has dimensions of length.
        eulerian_radius (bool): If true, the radius is assumed to be in 
                                Eulerian co-ordinates. Otherwise we assume
                                it is in Lagrangian co-ordinates.
        
    Returns:
        (float or array): Psi_r/r or Psi_r, same size as r

    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Unit test: test_process_radius_gives_radial_ratio
    """
    if quant_q is None:
        quant_q = Psi_q
    if isinstance(quant_q,list):
        quant_q_sum = np.sum(quant_q,0)
    else:
        quant_q_sum = quant_q
    if isinstance(Psi_q,list):
        Psi_q_sum = np.sum(Psi_q,0)
    else:
        Psi_q_sum = Psi_q
    if eulerian_radius:
        if taylor_expand:
            if (not isinstance(Psi_q,list)) or (not isinstance(quant_q,list)):
                raise Exception("Psi_q and quant_q must be lists" + 
                                " for computation of perturbative " + 
                                "eulerian radius ratio.")
            if order is None:
                raise Exception("Variable 'order' required for perturbative"
                                 + "calculation.")
            radial_ratio = get_eulerian_ratio_from_lagrangian_ratio(
                quant_q,Psi_q,order,expand_denom_only
            )
        else:
            if isinstance(quant_q,list):
                radial_ratio = [q/(1.0 + Psi_q_sum) for q in quant_q]
            else:
                radial_ratio = quant_q/(1.0 + Psi_q_sum)
    if radial_fraction:
        if eulerian_radius:
            return radial_ratio # Conversion to Eulerian co-ordinates.
        else:
            return quant_q
    else:
        if eulerian_radius:
            if isinstance(radial_ratio,list):
                return [r*val for val in radial_ratio]
            else:
                return r*radial_ratio
        else:
            if isinstance(quant_q,list):
                return [r*val for val in quant_q]
            else:
                return r*quant_q

def spherical_lpt_displacement(r,Delta,order=1,z=0,Om=0.3,
                               n2 = -1/143,nf1 = 5/9,
                               nf2 = 6/11,n3a = -4/275,n3b = -269/17875,
                               nf3a = 13/24,nf3b = 13/24,fixed_delta = False,
                               radial_fraction = False,S1r = None,
                               eulerian_radius=True,expand_denom_only=False,
                               taylor_expand=True,expand_euler_ratio=False,
                               return_all=False,
                               **kwargs):
    """
    Compute the radial component of the displacement field, in Lagrangian 
    perturbation theory, assuming spherical symmetry for the density field.
    
    Parameters:
        r (float or array): Radius/radii at which to compute the displacement
        Delta (scalar function): Final density field, as a function of r
        order (int): Order of expansion for LPT. Only 1,2, or 3 implemented
        z (float): Redshift at which to compute the displacement field.
        Om (float): Value of Omega_matter
        n2 (float): Exponent of Omega_m(z) used to approximate D_2(t)
        nf1 (float): Exponent of Omega_m(z) used to approximate f_1(t)
        nf2 (float): Exponent of Omega_m(z) used to approximate f_2(t)
        n3a (float): Exponent of Omega_m(z) used to approximate D_{3a}(t)
        n3b (float): Exponent of Omega_m(z) used to approximate D_{3b}(t)
        nf3a (float): Exponent of Omega_m(z) used to approximate f_{3a}(t)
        nf3b (float): Exponent of Omega_m(z) used to approximate f_{3b}(t)
        radial_fraction (bool): If True, compute Psi_r/r, rather than Psi_r 
        fixed_delta (bool): If True, Delta is assumed to be a pre-computed
                            array, rather than a function.
        eulerian_radius (bool): If True, radius is assumed to be in Eulerian
                                space. Otherwise (Default) it is assumed to be
                                in Lagrangian co-ordinates.
        expand_denom_only (bool): If True, expand only the denominator when
                                  converting to Eulerian co-ordinates
                                  (see processRadius)
        taylor_expand (bool): If True, use taylor-expanded final-density 
                              expression, rather than the non-perturbative
                              expression. Generally speaking this performs
                              less well, but we retain the option.
        expand_euler_ratio (bool): If True, use a Taylor expansion in 
                                   processRadius when converting to Eulerian
                                   co-ordinates.
        return_all (bool): If True, forces us to return separate velocity
                           corrections for each order.

    Returns:
        float or array: Radial component of the displacement field.

    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_spherical_lpt_displacement
    """
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    Delta_r = Delta if fixed_delta else Delta(r)
    split = (taylor_expand and eulerian_radius) or return_all
    # 1st order estimate of Psi_q:
    if S1r is None:
        S1r = get_S1r(Delta_r,1.0,Om,order=order,taylor_expand=taylor_expand,
                      **kwargs)
    Psi_q1 = get_psi_n_r(Delta_r,1.0,1,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,S1r=S1r,taylor_expand=taylor_expand,
                         **kwargs)
    Psi_q = np.zeros(Psi_q1.shape)
    Psi_q += Psi_q1
    if order == 1:
        Psi_arg = [Psi_q1] if split else Psi_q
        return process_radius(r,Psi_arg,radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,
                              order=order,taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 2nd order estimate of Psi_q:
    Psi_q2 = get_psi_n_r(Delta_r,1.0,2,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,S1r=S1r,taylor_expand=taylor_expand,
                         **kwargs)
    Psi_q += Psi_q2
    if order == 2:
        Psi_arg = [Psi_q1,Psi_q2] if split else Psi_q
        return process_radius(r,Psi_arg,radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 3rd order estimate of Psi_q:
    Psi_q3 = get_psi_n_r(Delta_r,1.0,3,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,S1r=S1r,taylor_expand=taylor_expand,
                         **kwargs)
    Psi_q += Psi_q3
    if order == 3:
        Psi_arg = [Psi_q1,Psi_q2,Psi_q3] if split else Psi_q
        return process_radius(r,Psi_arg,radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 4th order estimate of Psi_q:
    Psi_q4 = get_psi_n_r(Delta_r,1.0,4,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,S1r=S1r,taylor_expand=taylor_expand,
                         **kwargs)
    Psi_q += Psi_q4
    if order == 4:
        Psi_arg = [Psi_q1,Psi_q2,Psi_q3,Psi_q4] if split else Psi_q
        return process_radius(r,Psi_arg,radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 5th order estimate of Psi_q:
    Psi_q5 = get_psi_n_r(Delta_r,1.0,5,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,S1r=S1r,taylor_expand=taylor_expand,
                         **kwargs)
    Psi_q += Psi_q5
    Psi_arg = [Psi_q1,Psi_q2,Psi_q3,Psi_q4,Psi_q5] if split else Psi_q
    return process_radius(r,Psi_arg,radial_fraction=radial_fraction,
                          eulerian_radius=eulerian_radius,order=order,
                          taylor_expand=expand_euler_ratio,
                          expand_denom_only=expand_denom_only)

def spherical_lpt_velocity(r,Delta,order=1,z=0,Om=0.3,
                               n2 = -1/143,nf1 = 5/9,
                               nf2 = 6/11,n3a = -4/275,n3b = -269/17875,
                               nf3a = 13/24,nf3b = 13/24,h=1.0,
                               radial_fraction = False,fixed_delta = True,
                               eulerian_radius=True,taylor_expand=True,
                               expand_denom_only=False,expand_euler_ratio=False,
                               return_all=False,**kwargs):
    """
    Compute the radial component of the velocity field, in Lagrangian 
    perturbation theory, assuming spherical symmetry for the density field.
    
    Parameters:
        r (float or array): Radius/radii at which to compute the velocity. If
                            h = 1, assumes units of Mpc/h
        Delta (scalar function): Final density field, as a function of r
        order (int): Order of expansion for LPT. Only 1,2, or 3 implemented
        z (float): Redshift at which to compute the velocity field.
        Om (float): Value of Omega_matter
        n2 (float): Exponent of Omega_m(z) used to approximate D_2(t)
        nf1 (float): Exponent of Omega_m(z) used to approximate f_1(t)
        nf2 (float): Exponent of Omega_m(z) used to approximate f_2(t)
        n3a (float): Exponent of Omega_m(z) used to approximate D_{3a}(t)
        n3b (float): Exponent of Omega_m(z) used to approximate D_{3b}(t)
        nf3a (float): Exponent of Omega_m(z) used to approximate f_{3a}(t)
        nf3b (float): Exponent of Omega_m(z) used to approximate f_{3b}(t)
        h (float): Dimensionless Hubble rate. Defaults to 1.0, which means we
                   assume distances are in Mpc/h. Otherwise, non-zero h means
                   distances are expected to be in Mpc.
        radial_fraction (bool): If True, compute v/r, rather than v
        fixed_delta (bool): If True, Delta is assumed to be a pre-computed
                            array, rather than a function.
        eulerian_radius (bool): If True, radius is assumed to be in Eulerian
                                space. Otherwise (Default) it is assumed to be
                                in Lagrangian co-ordinates.
        expand_denom_only (bool): If True, expand only the denominator when
                                  converting to Eulerian co-ordinates
                                  (see processRadius)
        taylor_expand (bool): If True, use taylor-expanded final-density 
                              expression, rather than the non-perturbative
                              expression. Generally speaking this performs
                              less well, but we retain the option.
        expand_euler_ratio (bool): If True, use a Taylor expansion in 
                                   processRadius when converting to Eulerian
                                   co-ordinates.
        return_all (bool): If True, forces us to return separate velocity
                           corrections for each order.
    
    Returns:
        float or array: Radial component of the velocity field.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression test: test_spherical_lpt_velocity
    """
    print(kwargs)
    print({"radial_fraction":radial_fraction,"fixed_delta":fixed_delta,
           "eulerian_radius":eulerian_radius,"taylor_expand":taylor_expand})
    if order not in [1,2,3,4,5]:
        raise Exception("Perturbation order invalid or not implemented.")
    # Setup needed variables:
    D1_val = D1(z,Om,**kwargs)
    Delta_r = Delta if fixed_delta else Delta(r)
    split = (taylor_expand and eulerian_radius) or return_all
    a = 1/(1+z)
    H = Hz(z,Om,h=h,**kwargs)
    Omz = Omega_z(z,Om,**kwargs)
    # Get first order correction, from the initial conditions:
    S1r = get_S1r(Delta_r,1.0,Om,order=order,n2=n2,n3a=n3a,n3b=n3b,
                  taylor_expand=taylor_expand,**kwargs)
    # Get displacement field relative to Lagrangian radius 
    # (needed to compute correct Eulerian radii):
    if taylor_expand and eulerian_radius:
        Psi_q = get_psi_n_r(Delta_r,1.0,order,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,S1r=S1r,return_all=True,
                         taylor_expand=taylor_expand,**kwargs)
    else:
        Psi_q = spherical_lpt_displacement(
            1.0,Delta,order=order,z=z,Om=Om,n2 = n2,nf1 = nf1,nf2 = nf2,
            n3a = n3a,n3b = n3b,nf3a = nf3a,nf3b = nf3b,
            fixed_delta = fixed_delta,radial_fraction = True,
            eulerian_radius=False,S1r=S1r,taylor_expand=taylor_expand,
            **kwargs
        )
    # 1st order estimate of v_r:
    f1 = Omz**nf1
    v_r1 = a*H*f1*D1_val*S1r
    v_r = np.zeros(v_r1.shape)
    v_r += v_r1
    if order == 1:
        vr_arg = [v_r1] if split else v_r
        return process_radius(r,Psi_q,quant_q=vr_arg,
                              radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 2nd order estimate of v_r:
    D2_val = get_D_coefficients(
        Om,order=2,return_all = False,**kwargs
    )[0]
    f2_val = 2*(Omz**nf2)
    S2r = get_S2r(Delta_r,1.0,Om,order=order,n2=n2,n3a=n3a,n3b=n3b,
                  S1r = S1r,taylor_expand=taylor_expand,**kwargs)
    v_r2 = a*H*f2_val*D2_val*S2r
    v_r += v_r2
    if order == 2:
        vr_arg = [v_r1,v_r2] if split else v_r
        return process_radius(r,Psi_q,quant_q=vr_arg,
                              radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 3rd order estimate of v_r:
    f3a = 3*(Omz**nf3a)
    f3b = 3*(Omz**nf3b)
    D3a_val, D3b_val = get_D_coefficients(
        Om,order=3,return_all = False,**kwargs
    )
    S3ar, S3br = get_S3r(Delta_r,1.0,Om,order=order,S1r=S1r,
                         taylor_expand=taylor_expand,**kwargs)
    v_r3 = a*H*f3a*D3a_val*S3ar + a*H*f3b*D3b_val*S3br
    v_r += v_r3
    if order == 3:
        vr_arg = [v_r1,v_r2,v_r3] if split else v_r
        return process_radius(r,Psi_q,quant_q=vr_arg,
                              radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 4th order estimate of v_r:
    all_D4 = get_D_coefficients(
        Om,order=4,return_all = False,**kwargs
    )
    all_S4 = get_S4r(
        Delta_r,1.0,Om,order=order,S1r=S1r,taylor_expand=taylor_expand,**kwargs
    )
    # EdS approximation of linear growth rates for each term:
    all_f4 = [4*f1 for _ in all_D4]
    v_r4 = a*H*np.sum([f*D*S for f, D, S in zip(all_f4,all_D4,all_S4)],0)
    v_r += v_r4
    if order == 4:
        vr_arg = [v_r1,v_r2,v_r3,v_r4] if split else v_r
        return process_radius(r,Psi_q,quant_q=vr_arg,
                              radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)
    # 5th order estimate of v_r:
    all_D5 = get_D_coefficients(
        Om,order=5,return_all = False,**kwargs
    )
    all_S5 = get_S5r(
        Delta_r,1.0,Om,order=order,S1r=S1r,taylor_expand=taylor_expand,**kwargs
    )
    # EdS approximation of linear growth rates for each term:
    all_f5 = [5*f1 for _ in all_D5]
    v_r5 = a*H*np.sum([f*D*S for f, D, S in zip(all_f5,all_D5,all_S5)],0)
    v_r += v_r5
    vr_arg = [v_r1,v_r2,v_r3,v_r4,v_r5] if split else v_r
    return process_radius(r,Psi_q,quant_q=vr_arg,
                              radial_fraction=radial_fraction,
                              eulerian_radius=eulerian_radius,order=order,
                              taylor_expand=expand_euler_ratio,
                              expand_denom_only=expand_denom_only)


def get_los_velocities_for_void(void_centre,void_radius,snap,rbins,
        cell_vols=None,tree=None,observer=None,n_threads=-1
    ):
    """
    Compute the component of the velocity of tracers relative to a void
    centre, along the line of sight to the centre of the void.
    
    Parameters:
        void_centre (array, 3 components): Centre of the void
        void_radius (float): Radius of the void
        snap (pynbody.snapshot): Simulation snapshot
        rbins: Radius bins 
        cell_vols (array): Array of Voronoi volume weights for each tracer.
                           Assumes equal weight if not supplied.
        tree (cKD Tree): KD Tree for snapshot. Generated if not supplied.
        observer (array): Position of observer in the simulation.
                          Assumed to be the centre of the box if not supplied.
        n_threads (int): Number of threads to use for KD tree search. Default
                         -1 (all threads)
    
    Returns:
        r_par (array): Parallel components of the displacement from void centre
        u_par (array): Parallel components of velocity relative to void centre
        u: Velocity relative to void centre
        disp: Displacement from void centre
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py:
        Regression test: test_get_los_velocities_for_void
    """
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    wrapped = np.all(snap['pos'] > 0) and np.all(snap['pos'] < boxsize)
    wrapped_pos = snap['pos'] if wrapped else snapedit.wrap(
        snap['pos'],boxsize
    )
    wrapped_centre = void_centre if wrapped else snapedit.wrap(
        void_centre,boxsize
    )
    if observer is None:
        # assume observer is in the box centre:
        observer = np.array([boxsize/2]*3)
    los_vector = tools.get_unit_vector(
        snapedit.unwrap(void_centre - observer,boxsize)
    )
    if tree is None:
        tree = scipy.spatial.cKDTree(wrapped_pos,boxsize=boxsize)
    indices_list = np.array(tree.query_ball_point(wrapped_centre,\
            rbins[-1]*void_radius,workers=n_threads),dtype=np.int64)
    disp = snapedit.unwrap(snap['pos'][indices_list,:] - void_centre,boxsize)
    r = np.array(np.sqrt(np.sum(disp**2,1)))
    velocities_v = snap['vel'][indices_list,:]
    if cell_vols is None:
        cell_vols = np.ones(len(snap))
    weights = cell_vols[indices_list]
    v_centre = np.average(velocities_v,axis=0,weights=weights)
    u = velocities_v - v_centre
    u_par = u @ los_vector
    r_par = disp @ los_vector
    return r_par, u_par, disp, u




def ap_parameter(z, Om, Om_fid, h=0.7, h_fid=0.7, **kwargs):
    """
    Compute the Alcock-Paczynski parameter ε(z), which quantifies geometric
    distortions due to miss-specified cosmological parameters.

    ε(z) = [H(z) D_A(z)] / [H_fid(z) D_A_fid(z)]

    Parameters:
        z (float): Redshift
        Om (float): Matter density in test cosmology
        Om_fid (float): Fiducial matter density
        h (float): Hubble constant for test cosmology
        h_fid (float): Hubble constant for fiducial cosmology

    Returns:
        float: Alcock-Paczynski distortion parameter ε(z)
    
    Tests:
        Tested in cosmology_inference_tests/test_cosmology_utils.py:
        Regression tests: test_ap_parameter_regression
        Unit tests: test_ap_parameter_basic
    """
    # Cosmologies:
    cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=100*h_fid, Om0=Om_fid)
    cosmo_test = astropy.cosmology.FlatLambdaCDM(H0=100*h, Om0=Om)
    # Hubble rates:
    Hz = 100 * h * np.sqrt(Om * (1 + z)**3 + 1.0 - Om)
    Hz_fid = 100 * h_fid * np.sqrt(Om_fid * (1 + z)**3 + 1.0 - Om_fid)
    # Angular diameter distances:
    Da = np.fmax(cosmo_test.angular_diameter_distance(z).value,1e-16)
    Da_fid = np.fmax(cosmo_fid.angular_diameter_distance(z).value,1e-16)
    return (Hz * Da) / (Hz_fid * Da_fid)

def void_los_velocity_ratio_1lpt(r,Delta,f1,**kwargs):
    """
    Compute the line-of-sight (LOS) peculiar velocity at a position 
    relative to a void center, and divided by H(z)/(1+z), based on linear theory

    Parameters:
        r (float or array): Radius at which to compute the velocity
        Delta (function): Cumulative density contrast profile Δ(r)
        f1 (float): Linear growth rate

    Returns:
        float or array: LOS velocity in km/s
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression tests: test_void_los_velocity_ratio_1lpt
    """
    return -(f1 / 3.0) * Delta(r)

def void_los_velocity_ratio_derivative_1lpt(r,Delta,delta,f1,**kwargs):
    """
    Compute the derivative of the los velocity with respect to log(r),
    divided by H(z)/(1+z), using the linear 1LPT model. 
    
    Parameters:
        r (float or array): Radius at which to compute the velocity
        Delta (function): Cumulative density profile Δ(r)
        delta (function): Local density contrast δ(r)
        f1 (float or None): Growth rate

    Returns:
        float: Derivative of velocity w.r.t. log r
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression tests: test_void_los_velocity_ratio_derivative_1lpt
        Unit tests: test_lpt_velocity_derivative_integral
    """
    return f1*(Delta(r) - delta(r))

def void_los_velocity_ratio_semi_analytic(r,Delta,f1,params=None,
                                          exact_displacement=True,**kwargs):
    """
    Compute the line-of-sight (LOS) peculiar velocity at a position 
    relative to a void center, and divided by H(z)/(1+z),
    using a semi-analytic velocity model.

    Parameters:
        r (float or array): Radius at which to compute the velocity
        Delta (function): Cumulative density contrast profile Δ(r)
        f1 (float): Linear growth rate
        alphas (array): Coefficients of the quadratic and higher terms in the 
                        displacement field. Assumed to be in order from n = 2 
                        to n = N
        exact_displacement (bool): If True (default), use an exact expression 
                                   for the displacement field. Otherwise, 
                                   approximate it using the 1 LPT expression.
        params (list or None): Parameters for the model:

    Returns:
        float or array: LOS velocity in km/s
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression tests: test_void_los_velocity_ratio_semi_analytic
        Unit tests: test_semi_analytic_velocity_derivative_integral,
                    test_semi_analytic_model_functions_consistency
    """
    # Displacement field over r:
    Psi_r = 1 - np.cbrt(1 + Delta(r)) if exact_displacement else -Delta(r)/3
    # Extra terms:
    alphas = [] if params is None else params
    N = len(alphas) + 1
    sum_term = np.sum([alphas[n-2]*Psi_r**n for n in range(2,N+1)],0)
    return f1*(Psi_r + sum_term)

def void_los_velocity_ratio_derivative_semi_analytic(
        r,Delta,delta,f1,params=None,
        exact_displacement=True,**kwargs
    ):
    """
    Compute the derivative of the los velocity with respect to r_par,
    multiplied by H(z)/(1+z), using a semi-analytic velocity model.
    
    Parameters:
        Delta (function): Cumulative density profile Δ(r)
        delta (function): Local density contrast δ(r)
        r_par (float): LOS distance
        r_perp (float): Transverse distance
        f1 (float or None): Growth rate
        params (list or None): Parameters for the model.

    Returns:
        float: Derivative of velocity w.r.t. r_par
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py:
        Regression tests: test_void_los_velocity_ratio_derivative_semi_analytic
        Unit tests: test_semi_analytic_velocity_derivative_integral
    """
    Psi_r = 1 - np.cbrt(1 + Delta(r)) if exact_displacement else -Delta(r)/3
    dPsi_r_dlogr = ((Delta(r) - delta(r))/(np.cbrt(1 + Delta(r))**2) 
                    if exact_displacement else (Delta(r) - delta(r)))
    # Parameter extraction:
    alphas = [] if params is None else params
    N = len(alphas) + 1
    sum_term = np.sum([n*alphas[n-2]*Psi_r**(n-1) for n in range(2,N+1)],0)
    return f1*(1.0 + sum_term)*dPsi_r_dlogr

def void_los_velocity(z, Delta, r_par, r_perp, Om, f1=None,h=1,
                      vel_model = void_los_velocity_ratio_1lpt,
                      vel_params=None,**kwargs):
    """
    Compute the line-of-sight (LOS) peculiar velocity at a position 
    relative to a void center, based on a given model.

    Parameters:
        z (float): Redshift
        Delta (function): Cumulative density contrast profile Δ(r)
        r_par (float or array): LOS distance from void center
        r_perp (float or array): Transverse distance from void center
        Om (float): Matter density
        f1 (float or None): Growth rate. If None, computed via f_lcdm()
        h (float): Dimensionless Hubble rate. Set to h = 1 if using Mpc/h
                   for distances.                   
        vel_model (function): function which returns the dimensionless 
                            velocity ratio, (1+z)*v/(H(z)*r) at radius r. Should
                            have r, Delta, and f1 as parameters, but additional
                            parameters may be required depending on the model,
                            passed via kwargs.
        vel_params (array or None): Additional parameter for the velocity model

    Returns:
        float or array: LOS velocity in km/s
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression tests: test_void_los_velocity_regression
    """
    if f1 is None:
        f1 = f_lcdm(z, Om, **kwargs)
    # Cosmology dependent pre-factor:
    a = 1/(1+z)
    hz = Hz(z,Om,h=h,**kwargs)
    # Radius:
    r = np.sqrt(r_par**2 + r_perp**2)
    return a*hz*vel_model(r, Delta, f1, params=vel_params,**kwargs) * r_par

def get_dudr_hz_o1pz(Delta, delta, r_par, r_perp, f1, 
                     vel_model = void_los_velocity_ratio_1lpt,
                     dvel_dlogr_model = void_los_velocity_ratio_derivative_1lpt,
                     vel_params=None,**kwargs):
    """
    Compute the derivative of the los velocity with respect to r_par,
    multiplied by H(z)/(1+z). The advantage of this is that it is free from
    explicit dependence on z and Om (only depending on these via the 
    void profiles, \Delta(r) and \delta(r)).
    
    Parameters:
        Delta (function): Cumulative density profile Δ(r)
        delta (function): Local density contrast δ(r)
        r_par (float): LOS distance
        r_perp (float): Transverse distance
        f (float or None): Growth rate
        vel_model (function): Velocity ratio model, (same as in 
                              void_los_velocity)
        dvel_dlogr_model (function): Derivative of the dimensionless velocity 
                                     ratio model with respect to log(r). Ie, 
                                     should return d( v*(1+z)/(r*H(z)) )/dlog(r)
        vel_params (array or None): Additional parameter for the velocity model
        
    Returns:
        float: Derivative of velocity w.r.t. r_par
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression tests: test_get_dudr_hz_o1pz
    """
    r = np.sqrt(r_par**2 + r_perp**2)
    # Get the dimensionless velocity ratio, v*(1+z)/(r*H(z))
    vr_or = vel_model(r, Delta, f1, params = vel_params,**kwargs)
    # Get the logarithmic r derivative of the dimensionless velocity ratio,
    # d( v*(1+z)/(r*H(z)) )/dlog(r):
    dvr_or_dlogr = dvel_dlogr_model(r,Delta,delta,f1,params=vel_params,**kwargs)
    r_par_or = ratio_where_finite(r_par,r,undefined_value = 0.0)
    return vr_or + dvr_or_dlogr * (r_par_or)**2

def void_los_velocity_derivative(z, Delta, delta, r_par, r_perp, Om, f1=None, 
                                 reg_factor = 1e-12, vel_params=None,**kwargs):
    """
    Compute the derivative of the LOS velocity with respect to r_par.

    Assumes Delta(r) is the cumulative density contrast profile (mass within 
    a sphere), and delta(r) is the local density contrast (mass in a shell).
    These are related via integration, but must both be provided as functions.

    Parameters:
        z (float): Redshift
        Delta (function): Cumulative density profile Δ(r)
        delta (function): Local density contrast δ(r)
        r_par (float): LOS distance
        r_perp (float): Transverse distance
        Om (float): Matter density
        f1 (float or None): Growth rate (Lambda-CDM value assumed if not 
                            provided)
        vel_params (array or None): Additional parameter for the velocity model
        
    Returns:
        float: Derivative of velocity w.r.t. r_par
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression tests: test_void_los_velocity_derivative
    """
    hz = Hz(z, Om, **kwargs)
    if f1 is None:
        f1 = f_lcdm(z, Om, **kwargs)
    return (hz / (1.0 + z))*get_dudr_hz_o1pz(Delta,delta,r_par,r_perp,f1,
                                             vel_params=vel_params)


def z_space_jacobian(Delta, delta, r_par, r_perp, f1=None, z = 0, Om=0.3,
                     linearise_jacobian=False, vel_params=None,**kwargs):
    """
    Compute the Jacobian for the transformation from real to redshift space.

    Parameters:
        
        Delta (function): Cumulative density profile
        delta (function): Local density profile
        r_par (float): LOS distance
        r_perp (float): Transverse distance
        f1 (float or None): Linear growth rate. If None, computed using 
                            Lambda-CDM.
        z (float): Redshift
        Om (float): Matter density
        linearise_jacobian (bool): If True, return 1st-order approximation
        vel_params (array or None): Additional parameter for the velocity model

    Returns:
        float: Jacobian of the transformation
    Tests:
        Tested in cosmology_inference_tests/text_z_space_profile.py
        Regression tests: test_z_space_jacobian_regression,
                          test_z_space_jacobian_finite
        Unit tests: test_z_space_jacobian_positive, 
    """
    if f1 is None:
        f1 = f_lcdm(z, Om, **kwargs)
    dudr_hz_o1pz = get_dudr_hz_o1pz(
        Delta,delta,r_par,r_perp,f1,vel_params=vel_params,**kwargs
    )
    if linearise_jacobian:
        return 1.0 - dudr_hz_o1pz
    else:
        return 1.0 / (1.0 + dudr_hz_o1pz)


def to_z_space(r_par, r_perp, z=0, Om=0.3, Delta=None, u_par=None, f1=None, 
               vel_model = void_los_velocity_ratio_1lpt,**kwargs):
    """
    Transform real-space LOS coordinates to redshift space.

    If u_par is not supplied, a linear-theory velocity model is used
    based on the cumulative density profile Δ(r). Otherwise, uses 
    the explicitly provided velocity field u_par.

    Parameters:
        r_par (float or array): LOS coordinate in real space
        r_perp (float or array): Transverse coordinate in real space
        z (float): Redshift
        Om (float): Matter density
        Delta (function): Cumulative density profile Δ(r)
        u_par (float or array): LOS dimensionless velocity ratio (optional)
                                ie, (1+z)*v/(H(z)*r)
        f1 (float or None): Growth rate (Lambda-CDM value assumed if not 
                            provided)
        vel_model (function): function which returns the dimensionless 
                            velocity ratio, (1+z)*v/(H(z)*r) at radius r. Should
                            have r, Delta, and f1 as parameters, but additional
                            parameters may be required depending on the model,
                            passed via kwargs.

    Returns:
        list: [s_par, s_perp] — redshift-space coordinates
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression: test_to_z_space_regression
        Unit: test_to_real_space_consistency
    """
    if u_par is None:
        r = np.sqrt(r_par**2 + r_perp**2)
        u_par = vel_model(r, Delta, f1, **kwargs) * r_par
    s_par = r_par + u_par
    s_perp = r_perp
    return [s_par, s_perp]


def iterative_zspace_inverse_scalar(s_par, s_perp, f1, Delta, N_max = 5, 
                                    atol=1e-5, rtol=1e-5,
                                    vel_model = void_los_velocity_ratio_1lpt,
                                    vel_params = None,**kwargs):
    """
    Numerically invert the redshift-space LOS coordinate to estimate
    the real-space coordinate (r_par), assuming the linear-theory model.

    Parameters:
        s_par (float): LOS coordinate in redshift space
        s_perp (float): Perpendicular co-ordinate in refshift space
        f (float): Linear growth rate
        Delta (function): Cumulative density profile
        N_max (int): Maximum number of iterations
        atol (float): Absolute tolerance for convergence
        rtol (float): Relative tolerance for convergence
        vel_model (function): function which returns the dimensionless 
                            velocity ratio, (1+z)*v/(H(z)*r) at radius r. Should
                            have r, Delta, and f1 as parameters, but additional
                            parameters may be required depending on the model,
                            passed via kwargs.
        
    
    Returns:
        float: Estimated r_par
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression: test_iterative_zspace_inverse_scalar_regression
        Unit: test_iterative_zspace_inverse_scalar_identity,
              test_iterative_zspace_inverse_array_matches_scalar
    """
    r_par_guess = s_par
    r_perp = s_perp
    for _ in range(N_max):
        r = np.sqrt(r_par_guess**2 + r_perp**2)
        r_par_new = s_par / (1.0 + vel_model(
                                             r,Delta,f1,params=vel_params,
                                             **kwargs
                                   )
                            )
        if np.abs(r_par_new - r_par_guess) < atol or \
           np.abs(r_par_new / r_par_guess - 1.0) < rtol:
            break
        r_par_guess = r_par_new
    return r_par_guess

def iterative_zspace_inverse(s_par, s_perp, f1, Delta, N_max=5, atol=1e-5, 
                             rtol=1e-5,vel_params=None,**kwargs):
    """
    Array-compatible wrapper for iterative_zspace_inverse_scalar.

    Parameters:
        s_par (float or np.ndarray): LOS redshift-space coordinate(s)
        s_perp (float or np.ndarray): Perpendicular coordinate(s)
        f1 (float): Linear growth rate
        Delta (function): Cumulative density profile
        N_max (int): Max iterations
        atol (float): Absolute convergence tolerance
        rtol (float): Relative convergence tolerance
        vel_params (array or None): Additional parameter for the velocity model

    Returns:
        np.ndarray or float: Estimated real-space LOS coordinate(s)
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression: test_iterative_zspace_inverse_regression
        Unit: test_iterative_zspace_inverse_array_matches_scalar
    """
    s_par = np.asarray(s_par)
    s_perp = np.asarray(s_perp)
    if s_par.shape != s_perp.shape:
        raise ValueError("s_par and s_perp must have the same shape")
    # Scalar input case
    if s_par.ndim == 0:
        return iterative_zspace_inverse_scalar(s_par, s_perp, f1, Delta,
                                               N_max=N_max, atol=atol, 
                                               rtol=rtol,vel_params=vel_params,
                                               **kwargs)
    # Vectorized case (apply element-wise)
    return np.array([
        iterative_zspace_inverse_scalar(sp, sp_perp, f1, Delta,
                                        N_max=N_max, atol=atol, rtol=rtol,
                                        vel_params=vel_params,**kwargs)
        for sp, sp_perp in zip(s_par.flat, s_perp.flat)]
    ).reshape(s_par.shape)


def to_real_space(s_par, s_perp,
        Delta=None, u_par=None, f1=None, z=0, Om=0.3, N_max=5, atol=1e-5,
        rtol=1e-5, F_inv=None, vel_params=None, **kwargs
    ):
    """
    Convert redshift-space coordinates (s_par, s_perp) back into real-space 
    coordinates (r_par, r_perp), assuming either:

    - A linear-theory velocity model derived from the cumulative density 
        profile Delta(r), or
    - An explicit peculiar velocity field u_par

    Optionally uses a precomputed inverse mapping function F_inv, or falls 
    back to an iterative inversion method.

    Parameters:
        s_par (float): LOS redshift-space coordinate (must be scalar if 
                       inverting manually)
        s_perp (float or array): Transverse redshift-space coordinate
        z (float): Redshift
        Om (float): Matter density
        Om_fid (float or None): Fiducial matter density (unused here but may 
                                be passed downstream)
        Delta (function): Cumulative density contrast profile Δ(r), required 
                          for linear inversion
        u_par (float or array or None): LOS dimensionless peculiar velocity 
                                        ratio (if supplied, used directly). 
                                        Ie, (1+z)*v/(H(z)*r). Usually, this 
                                        isn't known, so leave as None to 
                                        compute it using the model.
        f1 (float or None): Growth rate; computed via f_lcdm if not supplied
        N_max (int): Max number of iterations for manual inversion
        atol (float): Absolute tolerance for iterative convergence
        rtol (float): Relative tolerance for iterative convergence
        F_inv (callable or None): Tabulated inverse mapping function
        vel_params (array or None): Additional parameter for the velocity model

    Returns:
        list: [r_par, r_perp] — real-space coordinates
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression: test_to_real_space_regression
        Unit: test_to_real_space_preserves_shape,
              test_to_real_space_consistency,
              test_z_space_profile_basic
    """
    r_perp = s_perp  # Perpendicular component is unaffected
    if u_par is None:
        # Use velocity model:
        if f1 is None:
            f1 = f_lcdm(z, Om, **kwargs)
        if F_inv is None:
            # Manual inversion via iterative method:
            if Delta is None:
                raise ValueError("Delta profile must be supplied for linear " + 
                                 "inversion.")
            # Use helper to handle the iterative logic
            r_par = iterative_zspace_inverse(s_par, s_perp, f1, Delta, N_max,
                                             atol = atol, rtol = rtol,
                                             vel_params=vel_params,**kwargs)
        else:
            # Use tabulated inverse function
            r_par = F_inv(s_par, s_perp)
    else:
        # Use directly supplied LOS peculiar velocity
        hz = Hz(z, Om, **kwargs)
        r_par = s_par - u_par
    return [r_par, r_perp]



def geometry_correction(s_par, s_perp, epsilon, **kwargs):
    """
    Apply the Alcock-Paczynski geometric correction to redshift-space 
    coordinates (s_par, s_perp) to account for a misspecified cosmology.

    This transformation rescales both the distance and the apparent angle 
    between line-of-sight (LOS) and transverse components, correcting the 
    apparent shape of cosmological voids distorted due to miss-specification
    of the cosmological parameters

    Based on the transformation:
        - s_factor adjusts the total distance accounting for ε-dependent 
          anisotropy.
        - mus (cosine of LOS angle) is adjusted to mus_new.
        - New coordinates are computed from these corrected values.

    Parameters:
        s_par (float or array): LOS redshift-space coordinate
        s_perp (float or array): Transverse redshift-space coordinate
        epsilon (float or None): Alcock-Paczynski distortion parameter ε
                                 (set to 1.0 if None)

    Returns:
        tuple: (s_par_new, s_perp_new) — corrected redshift-space coordinates
    
    Tests:
        Tested in cosmology_inference_tests/test_geometry.py
        Regression: test_geometry_correction_regression
        Unit: test_geometry_correction_identity,
              test_geometry_correction_output_shape,
              test_geometry_correction_reversibility
    """
    if epsilon is None:
        epsilon = 1.0
    # Radial distance from void center
    s = np.sqrt(s_par**2 + s_perp**2)
    # Cosine of LOS angle
    mus = ratio_where_finite(s_par,s,undefined_value=0.0)
    # Distance correction factor
    # This accounts for distortions in radial vs. transverse directions
    mus_rec = ratio_where_finite(1.0,mus,undefined_value=np.inf)
    s_factor = np.sqrt(1.0 + epsilon**2 * (mus_rec**2 - 1.0))
    # Apply AP correction to total distance and angle
    s_new = product_where_finite(s * mus * epsilon**(-2.0 / 3.0), s_factor,
                                 undefined_value = 0.0)
    # Corrected cosine of angle
    mus_new = ratio_where_finite(np.sign(mus) , s_factor)
    # Compute corrected coordinates
    s_par_new = mus_new * s_new
    s_perp_new = np.sign(s_perp) * s_new * np.sqrt(1.0 - mus_new**2)
    return s_par_new, s_perp_new


def z_space_profile(s_par, s_perp, rho_real, Delta, delta, f1=None,z=0, Om=0.3,
                    Om_fid=0.3111, epsilon=None, apply_geometry=False, 
                    vel_params=None,**kwargs):
    """
    Compute the redshift-space density profile ρ(s_par, s_perp) based on
    a real-space density profile and a model of redshift-space distortions.

    This includes:
      - Optional Alcock-Paczynski geometric correction (applied to input coords)
      - Coordinate transformation from redshift space to real space
      - Jacobian of the transformation (density scaling)
      - Evaluation of the density field at the mapped coordinates

    Parameters:
        s_par (float): LOS redshift-space coordinate
        s_perp (float): Transverse redshift-space coordinate
        rho_real (function): Real-space density profile ρ(r)
        z (float): Redshift
        Om (float): Matter density parameter of the assumed cosmology
        Delta (function): Cumulative density contrast profile Δ(r)
        delta (function): Local density contrast profile δ(r)
        Om_fid (float): Fiducial matter density (default: 0.3111)
        epsilon (float or None): Alcock-Paczynski distortion parameter ε.
                                 If None and apply_geometry is True, it is 
                                 computed.
        apply_geometry (bool): Whether to apply the AP correction
        vel_params (array or None): Additional parameter for the velocity model

    Returns:
        float: Redshift-space density ρ(s_par, s_perp)
    
    Tests:
        Tested in cosmology_inference_tests/test_z_space_profile.py
        Regression: test_z_space_profile_regression
        Unit: test_z_space_profile_basic
    """
    # Step 1: Apply Alcock-Paczynski geometric correction (if requested)
    if apply_geometry:
        if epsilon is None:
            epsilon = ap_parameter(z, Om, Om_fid, **kwargs)
        s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon)
    else:
        s_par_new = s_par
        s_perp_new = s_perp
    # Step 2: Convert redshift-space coords back to real-space coords
    r_par, r_perp = to_real_space(s_par_new, s_perp_new, f1=f1, z=z, Om=Om, 
                                  Delta=Delta,vel_params=vel_params, **kwargs)
    # Step 3: Compute the Jacobian of the transformation (∂r/∂s)
    jacobian = z_space_jacobian(Delta, delta, r_par, r_perp, Om=Om,z=z,f1=f1,
                                vel_params=vel_params,**kwargs)
    # Step 4: Evaluate the real-space profile at the recovered radius
    r = np.sqrt(r_par**2 + r_perp**2)
    return rho_real(r) * jacobian

#-------------------------------------------------------------------------------
# LIKELIHOOD AND POSTERIOR COMPUTATION


# Compute the covariance matrix of a set of data:
def covariance(data):
    """
    Compute the covariance of the supplied data. data should be an (N,M) 
    array of N data elements, each with M values.
    We seek to compute an M x M covariance matrix of this data
    
    Old method, and probably suitable for being deprecated.
    
    Parameters:
        data (array): NxM matrix array of N data samples, each of which is
                      an M-component vector.
    
    Returns:
        array: MxM covariance matrix of the data.
    
    Tests:
        Tested in test_covariance_and_statistics.py
        Regression tests: test_covariance_regression
        Unit tests: test_covariance_symmetry,
    """
    N, M = data.shape
    mean = np.mean(data,0)
    diff = data.T - mean[:,None]
    cov = np.matmul(diff,diff.T)/N
    return cov

def profile_jackknife_covariance(data,profile_function,*args,**kwargs):
    """
    Estimate covariance of the profile using the jackknife method.
    
    Parameters:
        data (array): NxM matrix array of N data samples, each of which is
                      an M-component vector.
        profile_function (function): Function which takes the data and fits
                                     a profile to it.
        *args (tuple): Additional arguments passed to the profile function.
        kwargs (dictionary): Keyword arguments passed to the profile function.
    
    Results:
        array: Covariance of the profile. 
    
    Tests:
        Tested in test_covariance_and_statistics.py
        Regression tests: test_profile_jackknife_covariance_regression
        Unit tests: test_profile_jackknife_covariance_shape
    """
    N, M = data.shape
    jacknife_data = np.zeros((N,M))
    for k in range(0,N):
        sample = np.setdiff1d(range(0,N),np.array([k]))
        jacknife_data[k,:] = profile_function(data[sample,:],*args,**kwargs)
    return covariance(jacknife_data)/(N-1)


def compute_singular_log_likelihood(x,Umap,good_eig):
    """
    Compute the log-likelihood of data after regularising poorly conditioned
    eigenvectors in the covariance matrix.
    
    Parameters:
        x (array): M-component residual of the data vector relative to a model.
        Umap (array): N x M matrix which maps the data residual into an 
                      eigenspace with well-conditioned eigenvectors. 
                      This is a projection operator constructed from the 
                      well-conditioned eigenvalues, while projecting out
                      everything in the poorly-conditioned subspace.
        good_eig (array): N-component array containing the well-conditioned 
                          eigenvalues in question.
        
    Returns:
        float: Log-likelihood after removing singular directions in data space.
    
    Tests:
        Tested in test_covariance_and_statistics.py
        Regression tests: test_compute_singular_log_likelihood_regression
        Unit tests: test_compute_singular_log_likelihood_basic
    """
    u = np.matmul(Umap,x) # Data mapped into the well-conditioned eigenspace
    Du = u/good_eig # Apply covariance, which is diagonal in this eigenspace
    uDu = np.sum(u*Du) # Covariance including only good eigenspace.
    N = len(good_eig)
    return -0.5*uDu - (N/2)*np.log(2*np.pi) - 0.5*np.sum(np.log(good_eig))

def get_tabulated_inverse(
        s_par,
        s_perp,
        ntab,
        Delta_func,
        f1,
        vel_model = void_los_velocity_ratio_1lpt,
        vel_params=None,
        use_iterative = True,
        **kwargs
    ):
    """
    Construct an interpolated function that inverts the mapping between real 
    space and redshift space.
    
    Parameters:
        s_par (array): Redshift space co-ordinates parallel to LOS
        s_perp (array): Redshift space co-ordinates perpendicular to LOS
        ntab (int): Number of grid points for the interpolation grid.
        vel_model (function): function which returns the dimensionless 
                            velocity ratio, (1+z)*v/(H(z)*r) at radius r. Should
                            have r, Delta, and f1 as parameters, but additional
                            parameters may be required depending on the model,
                            passed via kwargs.
        Delta_func (function) function that returns the cumulative density 
                              contrast at radius r.
        f1 (float): Linear growth rate.
        vel_params (array or None): Additional parameters for velocity model.
        use_iterative (bool): If True, use iterative method to invert. Otherwise
                              use generic fsolve approach
        
    Returns:
        Inverse function
    
    Tests:
        Tested in cosmology_inference/test_likelihood_and_posterior.py
        Regression tests: test_get_tabulated_inverse
        Unit tests: test_tabluated_inverse_accuracy
    """
    spar_vals = np.linspace(np.min(s_par),np.max(s_par),ntab)
    rperp_vals = np.linspace(np.min(s_perp),np.max(s_perp),ntab)
    rpar_vals = np.zeros((ntab,ntab))
    for i in range(0,ntab):
        for j in range(0,ntab):
            if use_iterative:
                rpar_vals[i,j] = iterative_zspace_inverse_scalar(
                    spar_vals[i], rperp_vals[j], f1, Delta_func, 
                                    vel_model = vel_model,
                                    vel_params = vel_params,**kwargs
                )
            else:
                F = (lambda rpar: rpar + rpar*vel_model(
                            np.sqrt(
                                rpar**2 + rperp_vals[j]**2),Delta_func,f1,
                                vel_params=vel_params,**kwargs
                            ) - spar_vals[i]
                    )
                rpar_vals[i,j] = scipy.optimize.fsolve(F,spar_vals[i])
    F_inv = lambda x, y: scipy.interpolate.interpn(
                                (spar_vals,rperp_vals),rpar_vals,
                                np.vstack((x,y)).T,method='cubic'
                            )
    return F_inv

def log_likelihood_aptest(theta, data_field, scoords, inv_cov, z,
                                   Delta, delta, rho_real,
                                   data_filter=None,
                                   cholesky=False,
                                   normalised=False,
                                   tabulate_inverse=False,
                                   ntab=10,
                                   sample_epsilon=False,
                                   Om_fid=0.3,
                                   singular=False,
                                   Umap=None,
                                   good_eig=None,
                                   F_inv=None,
                                   log_density=False,
                                   infer_profile_args=False,
                                   N_prof = 6,
                                   N_vel = 0,
                                   **kwargs):
    """
    Compute the log-likelihood for a cosmological + void profile model 
    using redshift-space distortion data and the Alcock-Paczynski test.

    Supports both:
    - Fixed profile functions (Δ(r), δ(r), ρ(r)), or
    - Parameterised profiles where shape parameters are inferred from data

    Parameters:
        theta (array): Model parameters:
            - [Om, f, ...] if sampling cosmology and profile
            - [epsilon, f, ...] if sample_epsilon=True
        data_field (array): Observed or simulated redshift-space density field
        scoords (array): (s_par, s_perp) coordinates
        inv_cov (array or Cholesky factor): Covariance inverse or lower triangle
        z (float): Redshift
        Delta, delta, rho_real (functions or constructors):
            - If infer_profile_args=True, must accept (*params)
            - Else, must be callables of r only
        data_filter (array or None): Optional data mask/filter
        cholesky (bool): If True, use Cholesky decomposition of covariance
        normalised (bool): Indicates whether to normalise the model
        tabulate_inverse (bool): If True, generate tabulated inverse map
        ntab (int): Number of samples for inverse interpolation
        sample_epsilon (bool): If True, sample ε instead of Om
        Om_fid (float): Fiducial Om used to compute ε
        singular (bool): Use projected likelihood in reduced eigenspace
        Umap (matrix): Projection matrix (for singular=True)
        good_eig (array): Eigenvalues for singular projection
        F_inv (callable or None): Precomputed inverse mapping
        log_density (bool): If True, use log(ρ) in likelihood
        infer_profile_args (bool): If True, extract profile params from θ
        N_prof (int): Number of density profile parameters
        N_vel (int): Number of velocity profile parameters

    Returns:
        float: Log-likelihood value
    
    Tests:
        Tested in cosmology_inference/test_likelihood_and_posterior.py
        Regression tests: test_log_likelihood_aptest_regression
    """
    # Apply optional data filtering
    if data_filter is not None:
        data_field = data_field[data_filter]
        scoords = scoords[data_filter, :]
        if cholesky or not singular:
            inv_cov = inv_cov[data_filter, :][:, data_filter]
    s_par, s_perp = scoords[:, 0], scoords[:, 1]
    # Unpack parameter vector
    if sample_epsilon:
        epsilon, f1 = theta[0], theta[1]
        Om = Om_fid
    else:
        Om, f1 = theta[0], theta[1]
        epsilon = ap_parameter(z, Om, Om_fid, **kwargs)
    profile_params = theta[2:(2 + N_prof)]
    vel_params = (None if N_vel == 0 
                  else theta[(2 + N_prof):(2 + N_prof + N_vel)])
    # Apply geometric correction to account for miss-specified cosmology. 
    # NB, this means that we SHOULDN'T apply geometry corrections below, 
    # because they have already been applied here!
    s_par_new, s_perp_new = geometry_correction(s_par,s_perp,epsilon)
    # Construct profile functions
    if infer_profile_args:
        Delta_func = lambda r: Delta(r, *profile_params)
        delta_func = lambda r: delta(r, *profile_params)
        rho_func = lambda r: rho_real(r, *profile_params)
    else:
        Delta_func = Delta
        delta_func = delta
        rho_func = rho_real
    # Generate a tabulated inverse mapping if needed
    if F_inv is None and tabulate_inverse:
        F_inv = get_tabulated_inverse(
            s_par_new,s_perp_new,ntab,Delta_func,f1,vel_params=vel_params,
            **kwargs
        )
    # Evaluate the model at each (s_par, s_perp) coordinate. Setting 
    # apply_geometry to False, because they were already applied above.
    model_field = z_space_profile(
        s_par_new, s_perp_new, rho_func, Delta_func, delta_func,f1=f1,
        z=z, Om=Om, epsilon=epsilon,apply_geometry=False,F_inv=F_inv,
        vel_params=vel_params,**kwargs
    )
    if log_density:
        model_field = np.log(model_field)
    # Project into reduced space if using singular-mode filtering
    if singular:
        data_field = Umap @ data_field
        model_field = Umap @ model_field
        inv_cov = np.diag(1.0 / good_eig)
    # Compute residual
    delta_vec = (1.0 - model_field/data_field if normalised else 
                 data_field - model_field)
    if cholesky:
        alpha = scipy.linalg.solve_triangular(inv_cov, delta_vec, lower=True)
        return -0.5 * np.dot(alpha, alpha)
    else:
        return -0.5 * np.dot(delta_vec, inv_cov @ delta_vec)


# Un-normalised log prior:

def log_flat_prior_single(x, bounds):
    """
    Compute the log of a flat prior over a bounded interval.

    Parameters:
        x (float): Parameter value
        bounds (tuple): (min, max) bounds of the flat prior

    Returns:
        float: 0 if within bounds, -inf if out of bounds
    
    Tests:
        Tested in cosmology_inference/test_likelihood_and_posterior.py
        Regression tests: test_log_flat_prior_single
        Unit tests: test_log_flat_prior_single_inside,
                    test_log_flat_prior_single_outside
        
    """
    xmin, xmax = bounds
    return 0.0 if xmin <= x <= xmax else -np.inf

def log_flat_prior(theta, bounds):
    """
    Compute the joint log of a flat prior over a hyperrectangle.

    Parameters:
        theta (array-like): List of parameter values
        bounds (list of tuples): List of (min, max) bounds for each parameter

    Returns:
        float: 0 if all parameters are within bounds, -inf otherwise
    
    Tests:
        Tested in cosmology_inference/test_likelihood_and_posterior.py
        Regression tests: test_log_flat_prior
        Unit tests: test_log_flat_prior_batch_inside,
                    test_log_flat_prior_batch_outside
    """
    if any(t < b[0] or t > b[1] for t, b in zip(theta, bounds)):
        return -np.inf
    return 0.0

def log_probability_aptest(theta, *args, **kwargs):
    """
    Compute the log posterior for the AP test likelihood.

    Posterior is defined as:
        log_posterior = log_prior + log_likelihood

    Assumes a flat prior with hard bounds defined by 'theta_ranges' in kwargs.

    Parameters:
        theta (array): Parameter vector
        *args: Passed to log_likelihood_aptest_revised
        **kwargs:
            - theta_ranges (list of tuples): Prior bounds for each parameter
            - All other kwargs passed to log_likelihood_aptest

    Returns:
        float: Log posterior
    
    Tests:
        Tested in cosmology_inference/test_likelihood_and_posterior.py
        Regression tests: test_log_probability_aptest_regression
        Unit tests: test_log_probability_aptest_sanity
    """
    theta_ranges = kwargs.pop("theta_ranges", None)
    if theta_ranges is None:
        raise ValueError("Missing 'theta_ranges' in kwargs for " + 
                         "prior evaluation.")
    lp = log_flat_prior(theta, theta_ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_aptest(theta, *args, **kwargs)

#-------------------------------------------------------------------------------
# GAUSSIANITY TESTING

def tikhonov_regularisation(cov_matrix, lambda_reg=1e-10):
    """
    Apply Tikhonov regularisation to a covariance matrix.

    This adds a scaled identity matrix to the covariance, stabilising the 
    inverse:
        cov_reg = cov + alpha * I

    Parameters:
        cov_matrix (ndarray): Original covariance matrix (NxN)
        alpha (float): Regularisation strength (default: 1e-3)

    Returns:
        ndarray: Regularised covariance matrix
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics.py
        Regression tests: test_tikhonov_regularisation_regression
        Unit tests: test_tikhonov_regularisation_identity
    """
    return cov_matrix + lambda_reg * np.eye(cov_matrix.shape[0])

def regularise_covariance(cov, lambda_reg=1e-10):
    """
    Symmetrise and regularise a covariance matrix using Tikhonov regularisation.

    Ensures the matrix is symmetric and numerically stable for inversion.

    Parameters:
        cov (ndarray): Input covariance matrix
        lambda_reg (float): Tikhonov regularisation strength (default: 1e-10)

    Returns:
        ndarray: Symmetrised and regularised covariance matrix
    
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics
        Regression tests: test_regularise_covariance_regression
        Unit tests: test_regularise_covariance_is_symmetric,
                    test_regularise_covariance_is_positive_definite,
                    test_regularise_covariance_symmetry
    """
    symmetric_cov = 0.5 * (cov + cov.T)
    regularised_cov = tikhonov_regularisation(
        symmetric_cov, lambda_reg=lambda_reg
    )
    return regularised_cov


def get_inverse_covariance(cov, lambda_reg=1e-10):
    """
    Compute the inverse of a (regularised) covariance matrix using Cholesky 
    decomposition.

    This involves:
    - Symmetrising the input
    - Applying Tikhonov regularisation for numerical stability
    - Computing the inverse via Cholesky factorisation:
        C^(-1) = (L^(-1))^T @ L^(-1)

    Parameters:
        cov (ndarray): Covariance matrix (NxN)
        lambda_reg (float): Regularisation strength (default: 1e-10)

    Returns:
        ndarray: Inverse of the regularised covariance matrix
    
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics
        Regression tests: test_get_inverse_covariance_regression
        Unit tests: test_inverse_covariance_sane,
                    test_get_inverse_covariance_consistency
    """
    regularised_cov = regularise_covariance(cov, lambda_reg=lambda_reg)
    L = np.linalg.cholesky(regularised_cov)      # Lower triangular matrix
    P = np.linalg.inv(L)                         # Inverse of L
    inv_cov = np.matmul(P.T, P)                  # Reconstruct full inverse
    return inv_cov


def range_excluding(kmin, kmax, exclude):
    """
    Generate a list of integers in [kmin, kmax), excluding values in 'exclude'.

    Parameters:
        kmin (int): Start of range (inclusive)
        kmax (int): End of range (exclusive)
        exclude (list or array): Values to remove from the range

    Returns:
        ndarray: Array of integers not in 'exclude'
    
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics
        Unit tests: test_range_excluding_basic
    """
    return np.setdiff1d(range(kmin, kmax), exclude)

def get_nonsingular_subspace(C, lambda_reg,
                             lambda_cut=None,
                             normalised_cov=False,
                             mu=None):
    """
    Compute a projection onto the non-singular subspace of a covariance matrix.

    The covariance is regularised and optionally normalised before eigenvalue 
    decomposition. Only eigenvectors with eigenvalues above a cutoff are 
    retained.

    Parameters:
        C (ndarray): Covariance matrix (k x k)
        lambda_reg (float): Tikhonov regularisation parameter
        lambda_cut (float or None): Minimum eigenvalue to keep 
                                    (default: 10 * lambda_reg)
        normalised_cov (bool): If True, normalise C by mu.outer(mu)
        mu (ndarray or None): Mean vector (required if normalised_cov=True)

    Returns:
        tuple:
            - Umap (ndarray): Projection matrix to nonsingular eigenspace
            - good_eig (ndarray): Retained eigenvalues
    
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics
        Regression tests: test_compute_singular_log_likelihood_regression
        Unit tests: test_compute_singular_log_likelihood_basic,
                    test_get_nonsingular_subspace_structure
    """
    reg_cov = regularise_covariance(C, lambda_reg=lambda_reg)
    if normalised_cov:
        if mu is None:
            raise ValueError("Mean 'mu' must be provided for normalised " + 
                             "covariance.")
        norm_cov = C / np.outer(mu, mu)
        norm_reg_cov = regularise_covariance(norm_cov, lambda_reg=lambda_reg)
        eig, U = scipy.linalg.eigh(norm_reg_cov)
    else:
        eig, U = scipy.linalg.eigh(reg_cov)
    if lambda_cut is None:
        lambda_cut = 10 * lambda_reg
    good_eig = eig[eig >= lambda_cut]
    Umap = U[:, eig >= lambda_cut].T  # Each row is a retained eigenvector
    return Umap, good_eig


def get_solved_residuals(samples, covariance, xbar,
                         singular=False,
                         normalised_cov=False,
                         L=None,
                         lambda_cut=1e-23,
                         lambda_reg=1e-27,
                         Umap=None,
                         good_eig=None):
    """
    Compute whitened residuals for Gaussianity testing.

    Supports both full-rank and projected cases using a reduced eigenbasis.

    Parameters:
        samples (ndarray): Shape (k, n) — k variables, n samples
        covariance (ndarray): Covariance matrix (k x k)
        xbar (ndarray): Mean of the samples (length k)
        singular (bool): Whether to project into nonsingular subspace
        normalised_cov (bool): Whether to normalise by the mean
        L (ndarray or None): Cholesky factor of covariance (optional)
        lambda_cut (float): Cutoff for eigenvalue retention
        lambda_reg (float): Regularisation strength
        Umap (ndarray or None): Projection matrix (optional)
        good_eig (ndarray or None): Retained eigenvalues (optional)

    Returns:
        ndarray: Whitened residuals (projected or full)
    
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics
        Regression tests: test_get_solved_residuals
        Unit tests: test_compute_singular_log_likelihood_basic,
                    test_get_solved_residuals_shape
    """
    k, n = samples.shape
    if not singular:
        if normalised_cov:
            residual = samples / xbar[:, None] - 1.0
        else:
            residual = samples - xbar[:, None]
        if L is None:
            reg_cov = regularise_covariance(covariance, lambda_reg=lambda_reg)
            L = scipy.linalg.cholesky(reg_cov, lower=True)
        # Solve triangular system for each sample
        solved_residuals = np.array([
            scipy.linalg.solve_triangular(L, residual[:, i], lower=True)
            for i in tools.progressbar(range(n))
        ]).T
    else:
        # Project into reduced eigenspace
        if Umap is None or good_eig is None:
            Umap, good_eig = get_nonsingular_subspace(
                covariance, lambda_reg=lambda_reg,
                lambda_cut=lambda_cut,
                normalised_cov=normalised_cov,
                mu=xbar
            )
        if normalised_cov:
            residual = Umap @ (samples / xbar[:, None] - 1.0)
        else:
            residual = Umap @ (samples - xbar[:, None])
        solved_residuals = residual / np.sqrt(good_eig[:, None])
    return solved_residuals


def compute_normality_test_statistics(samples,
                                      covariance=None,
                                      xbar=None,
                                      solved_residuals=None,
                                      low_memory_sum=False,
                                      **kwargs):
    """
    Compute test statistics to evaluate Gaussianity of residuals.

    Statistics:
      - A: Skewness-like cubic term
      - B: Kurtosis-like quartic term

    Parameters:
        samples (ndarray): (k, n) array of samples
        covariance (ndarray or None): Covariance matrix (used if residuals not
                                      provided)
        xbar (ndarray or None): Mean vector (used if residuals not provided)
        solved_residuals (ndarray or None): Precomputed whitened residuals
        low_memory_sum (bool): If True, use a lower-memory summation loop

    Returns:
        list: [A, B] — test statistics
    
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics
        Regression tests: test_compute_normality_statistics_regression
        Unit tests: test_compute_normality_statistics_output
    """
    n = samples.shape[1]
    k = samples.shape[0]
    if covariance is None:
        covariance = np.cov(samples)
    if xbar is None:
        xbar = np.mean(samples, axis=1)
    if solved_residuals is None:
        solved_residuals = get_solved_residuals(
            samples, covariance, xbar, **kwargs
        )
    if low_memory_sum:
        Ai = np.array([
            np.sum(
                np.sum(
                    solved_residuals[:, i][:, None] * solved_residuals, axis=0
                ) ** 3
            ) for i in tools.progressbar(range(n))
        ])
        A = np.sum(Ai) / (6 * n)
    else:
        product = solved_residuals.T @ solved_residuals
        A = np.sum(product ** 3) / (6 * n)
    B = np.sqrt(n / (8 * k * (k + 2))) * (
        np.sum(np.sum(solved_residuals ** 2, axis=0) ** 2) / n - k * (k + 2)
    )
    return [A, B]


#-------------------------------------------------------------------------------
# STACKED DENSITY FIELDS

def get_zspace_centres(halo_indices, snap_list, snap_list_rev,
                       hrlist=None, recompute_zspace=False,
                       swapXZ=False, reverse=True):
    """
    Compute redshift-space void centers from a halo catalogue (in reverse 
    simulations).

    This function:
      - Loads the redshift-space positions of all particles in the forward 
        snapshot
      - Uses the halo catalogue (from reverse sim) to identify void particles
      - Computes the redshift-space center for each void
      - Applies remapping to shift into the final coordinate frame

    Parameters:
        halo_indices (list of lists): Per-snapshot list of halo indices 
                                      representing voids
        snap_list (list): Forward simulation snapshots (used for positions)
        snap_list_rev (list): Reverse snapshots (used for halos = voids)
        hrlist (list or None): If supplied, overrides halos loaded from 
                               snap_list_rev
        recompute_zspace (bool): If True, force recomputation of redshift-space 
                                 positions
        swapXZ (bool): Whether to swap X and Z axes during coordinate remapping
        reverse (bool): Whether to reflect coordinates around box center during 
                        remapping

    Returns:
        list of arrays: Redshift-space centers of voids for each snapshot
    
    Tests:
        Tested in test_stacking_functions.py
        Regression tests: test_get_zspace_centres
    """
    if len(halo_indices) != len(snap_list):
        raise ValueError("halo_indices list does not match snapshot list.")
    num_samples = len(halo_indices)
    # Preallocate list of centers, shape (N_voids, 3) per snapshot
    centres = [np.full((len(halos), 3), np.nan) for halos in halo_indices]
    for ns in range(num_samples):
        snap = snap_list[ns]
        # Get particle index sort order (needed for consistent indexing)
        if os.path.isfile(snap.filename + ".snapsort.p"):
            sorted_indices = tools.loadPickle(snap.filename + ".snapsort.p")
        else:
            sorted_indices = np.argsort(snap['iord'])
        # Load halos from reverse simulation (or use override list)
        halos = hrlist[ns] if hrlist is not None else snap_list_rev[ns].halos()
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        # Get redshift-space positions of all particles in forward snapshot
        positions = tools.loadOrRecompute(
            snap.filename + ".z_space_pos.p",
            simulation_tools.redshift_space_positions,
            snap,
            centre=np.array([boxsize / 2] * 3),
            _recomputeData=recompute_zspace
        )
        for k in tools.progressbar(range(len(halo_indices[ns]))):
            index = halo_indices[ns][k]
            if index >= 0:
                # Lookup particle indices in halo
                particle_ids = halos[index + 1]['iord']
                particle_pos = positions[sorted_indices[particle_ids], :]

                # Compute center in redshift space, apply remapping
                centre = context.computePeriodicCentreWeighted(
                    particle_pos, periodicity=boxsize
                )
                centres[ns][k, :] = tools.remapAntiHaloCentre(
                    centre,
                    boxsize,
                    swapXZ=swapXZ,
                    reverse=reverse
                )
    return centres

def combine_los_lists(los_lists):
    """
    Combine multiple LOS particle lists into a single per-void list.

    Each input in `los_lists` is expected to be a list of per-void arrays.
    The combined result takes the non-empty list for each void from the 
    most recent list that has valid data.

    NOTE: This function is unused and may be deprecated. Originally used to 
    combine multiple versions of LOS data (e.g., density vs velocity space).

    Parameters:
        los_lists (list of lists): Each element is a list of LOS arrays 
                                   (same length across inputs)

    Returns:
        list: Combined list of LOS arrays (same length as inner lists)

    Raises:
        Exception: If input lists are not all the same length
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Unit tests: test_combine_los_lists_basic
        Regression tests: test_combine_los_lists
        
    """
    lengths = np.array([len(x) for x in los_lists], dtype=int)
    if not np.all(lengths == lengths[0]):
        raise Exception("Cannot combine LOS lists with different lengths.")
    new_list = []
    # Build a matrix of particle counts for each void across the different lists
    num_parts = np.vstack([
        np.array([len(x) for x in los_list], dtype=int)
        for los_list in los_lists
    ]).T  # Shape: (N_voids, N_lists)
    to_use = -np.ones(lengths[0], dtype=int)  # Index of list to use per void
    for k in range(len(los_lists)):
        to_use[num_parts[:, k] > 0] = k  # Use the latest list with valid data
    for k in range(lengths[0]):
        if to_use[k] >= 0:
            new_list.append(los_lists[to_use[k]][k])
        else:
            new_list.append(np.zeros((0, 2)))  # Empty fallback
    return new_list


def get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, 
        stacked=True
    ):
    """
    Construct a stacked dataset of LOS positions around void centers, 
    normalized by effective void radius (R_eff).

    Parameters:
        los_pos (list): Per-void list of LOS particle positions 
                        (usually 2D: void x LOS array)
        spar_bins (array): Bins along the line of sight (s_parallel)
        sperp_bins (array): Bins perpendicular to LOS (s_perp)
        radii (list of arrays): Effective radii (R_eff) per void
        stacked (bool): If True, return a single combined array of all 
                        particles;
                        If False, return nested list of per-void normalized 
                        LOS data

    Returns:
        array or list: 
            - If stacked=True: ndarray of shape (N_particles, 2)
            - Else: List of per-void arrays of normalized LOS positions
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_2d_fields_per_void_regression,
                          test_get_2d_void_stack_from_los_pos_regression,
                          test_get_2d_field_from_stacked_voids_regression,
                          test_get_field_from_los_data_regression
        Unit tests: test_get_2d_void_stack_from_los_pos_shape,
                    test_get_field_from_los_data_basic,
                    test_get_2d_field_from_stacked_voids_basic,
                    test_get_field_from_los_data_shape
    """
    # Identify voids with at least one non-empty LOS
    voids_used = [np.array([len(x) for x in los]) > 0 for los in los_pos]
    # Remove empty LOS entries to avoid stacking issues
    los_pos_filtered = [[x for x in los if len(x) > 0] for los in los_pos]
    # Compute cell volumes (not used here directly, possibly for weighting 
    # later)
    # Filter radii accordingly
    void_radii = [rad[filt] for rad, filt in zip(radii, voids_used)]
    # Normalize LOS positions by R_eff (to work in R-scaled coordinates)
    los_list_reff = [
        [np.abs(los / rad) for los, rad in zip(all_los, all_radii)]
        for all_los, all_radii in zip(los_pos_filtered, void_radii)
    ]
    if stacked:
        # Flatten all voids into one large particle stack
        stacked_particles_reff = np.vstack(
            [np.vstack(los_list) for los_list in los_list_reff]
        )
        return stacked_particles_reff
    else:
        # Return per-void lists of normalized LOS positions
        return los_list_reff


def get_weights_for_stack(los_pos, void_radii, additional_weights=None, 
        stacked=True
    ):
    """
    Compute volume-based stacking weights for LOS particles.

    This function was originally designed to apply both:
      - A 1 / R_eff^3 scaling to account for void size (density normalization)
      - Optional signal-to-noise or custom weights

    In later refactors, these were separated into logically distinct steps.
    May be deprecated.

    Parameters:
        los_pos (list): List of LOS particle arrays per void
        void_radii (list): List of R_eff per void
        additional_weights (list or scalar or None): Optional multiplicative 
                                                     weights
        stacked (bool): If True, flatten the weights across all voids

    Returns:
        array or list of arrays:
            - Flattened array if stacked=True
            - Nested weight list matching los_pos structure if stacked=False
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_weights_for_stack
        Unit tests: test_get_weights_for_stack_basic
    """
    if additional_weights is None:
        v_weight = [
            [(1.0 / rad)**3 * np.ones(len(los))
             for los, rad in zip(all_los, all_radii)]
            for all_los, all_radii in zip(los_pos, void_radii)
        ]
    else:
        if isinstance(additional_weights, list):
            # Per-void weights (possibly per simulation)
            v_weight = [
                [(1.0 / rad)**3 * np.ones(len(los)) * weight
                 for los, rad, weight in zip(all_los, all_radii, all_weights)]
                for all_los, all_radii, all_weights in zip(
                    los_pos, void_radii, additional_weights
                )
            ]
        else:
            # Scalar weight broadcast across all
            v_weight = [
                [(1.0 / rad)**3 * np.ones(len(los)) * additional_weights
                 for los, rad in zip(all_los, all_radii)]
                for all_los, all_radii in zip(los_pos, void_radii)
            ]
    if stacked:
        return np.hstack([np.hstack(rad) for rad in v_weight])
    else:
        return v_weight


def get_field_from_los_data(los_data, spar_bins, sperp_bins, v_weight,
                            void_count, nbar=None):
    """
    Bin all LOS particles into a 2D histogram to compute the stacked void 
    density field.

    Includes a cylindrical Jacobian factor and optional normalization by cosmic 
    mean density.

    Parameters:
        los_data (ndarray): Array of shape (N_particles, 2) with (s_par, s_perp)
                            values
        spar_bins (array): Bin edges along LOS direction
        sperp_bins (array): Bin edges perpendicular to LOS
        v_weight (array): Weights for each particle (same length as los_data)
        void_count (int): Total number of contributing voids
        nbar (float or None): Mean number density. If supplied, normalizes 
                              output.

    Returns:
        ndarray: 2D stacked density field (shape: [N_spar_bins, N_sperp_bins])
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_field_from_los_data_regression
        Unit tests: test_get_field_from_los_data_basic,
                    test_get_field_from_los_data_shape
    """
    cell_volumes_reff = np.outer(np.diff(spar_bins), np.diff(sperp_bins))
    hist = np.histogramdd(
        los_data,
        bins=[spar_bins, sperp_bins],
        density=False,
        weights=v_weight / (2 * np.pi * los_data[:, 1])
    )[0]
    if nbar is not None:
        return hist / (2 * void_count * cell_volumes_reff * nbar)
    else:
        return hist / (2 * void_count * cell_volumes_reff)


def get_2d_fields_per_void(los_per_void, sperp_bins, spar_bins,
                           void_radii, nbar=None):
    """
    Compute 2D density fields for a set of voids from their LOS particle 
    positions.

    The stacking is performed in coordinates rescaled by void radius R_eff,
    so the resulting histograms are corrected to yield densities in physical 
    units.

    This involves:
      - Histogramming each void’s LOS particles into (s_perp, s_par) bins
      - Dividing by the Jacobian factor in cylindrical coordinates (2π * s_perp)
      - Applying a 1/R_eff^3 scaling to convert back to real (Mpc^3) density
      - Optionally normalizing by mean cosmic density nbar to yield 1 + δ

    Parameters:
        los_per_void (list): Per-void list of LOS particle positions, each 
                             shape (N, 2)
        sperp_bins (array): Radial bins (perpendicular to LOS)
        spar_bins (array): Bins along LOS
        void_radii (array): R_eff per void (same length as los_per_void)
        nbar (float or None): Mean particle density. If supplied, returns 
                              dimensionless density.

    Returns:
        ndarray: 3D array of shape (N_voids, N_spar_bins, N_sperp_bins) 
                 representing 2D density fields
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_2d_fields_per_void_regression
    """
    # Compute cell volumes in scaled (R_eff) units
    cell_volumes_reff = np.outer(np.diff(spar_bins), np.diff(sperp_bins))
    # Compute 2D density histograms for each void 
    # (weighted by Jacobian correction)
    histograms = np.array([
        np.histogramdd(los, bins=[spar_bins, sperp_bins],
                       density=False,
                       weights=1.0 / (2 * np.pi * los[:, 1]))[0]
        for los in los_per_void
    ])
    # Convert back to physical units (1 / R_eff^3 for scaling back from R_eff 
    # units)
    volume_weight = 1.0 / void_radii**3
    # Denominator accounts for volume and optional density normalization
    if nbar is not None:
        denominator = 2 * cell_volumes_reff * nbar
    else:
        denominator = 2 * cell_volumes_reff
    # Final normalized density: (1/R^3) * histogram / volume
    density = (
        volume_weight[:, None, None] * histograms / denominator[None, :, :]
    )
    return density


def get_2d_field_from_stacked_voids(los_per_void, sperp_bins, spar_bins,
                                    void_radii, weights=None, nbar=None):
    """
    Compute the average 2D density field from a collection of voids, 
    each represented by LOS particles.

    This function:
        - Computes a 2D density histogram for each void
        - Averages them (optionally weighted)

    Parameters:
        los_per_void (list): List of per-void particle LOS positions
        sperp_bins (array): Radial bins perpendicular to LOS
        spar_bins (array): Bins along the LOS direction
        void_radii (list): Effective radii (R_eff) for each void
        weights (array or None): Per-void weights for averaging (optional)
        nbar (float or None): Mean particle density for normalization (optional)

    Returns:
        ndarray: Averaged 2D density field
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_2d_field_from_stacked_voids_regression
        Unit tests: test_get_2d_field_from_stacked_voids_basic
    """
    # Compute per-void 2D density histograms
    density = get_2d_fields_per_void(
        los_per_void, sperp_bins, spar_bins, void_radii, nbar=nbar
    )
    # Average across voids (optionally weighted)
    return np.average(density, axis=0, weights=weights)

def profile_broken_power_log(r, A, r0, c1, f1, B):
    """
    Logarithmic version of a broken power-law void density profile.

    Used internally to simplify fitting and enforce positivity.

    Form:
        log(ρ/ρ̄) = log|A + B(r/r₀)² + (r/r₀)^4| + ((c₁ - 4)/f₁) * 
                    log(1 + (r/r₀)^f₁)

    Parameters:
        r (float or array): Radial distance
        A, B (float): Amplitude coefficients
        r0 (float): Characteristic radius
        c1, f1 (float): Shape parameters

    Returns:
        float or array: Logarithm of the void density profile
    
    Tests:
        Tested in cosmology_inference_tests/test_profiles.py
        Regression tests: test_profile_broken_power_log
        Unit tests: test_broken_power_log_consistency
    """
    return np.log(np.abs(A + B * (r / r0)**2 + (r / r0)**4)) + \
           ((c1 - 4) / f1) * np.log(1 + (r / r0)**f1)

def profile_broken_power(r, A, r0, c1, f1, B):
    """
    Broken power-law profile for void density.

    This is the exponential of profile_broken_power_log, ensuring a positive 
    density.

    Parameters:
        r (float or array): Radial distance
        A, B (float): Amplitude coefficients
        r0 (float): Characteristic radius
        c1, f1 (float): Shape parameters

    Returns:
        float or array: Density contrast profile δ(r)
    
    Tests:
        Tested in cosmology_inference_tests/test_profiles.py
        Regression tests: test_broken_power_regression
        Unit tests: test_broken_power_log_consistency
    """
    return np.exp(profile_broken_power_log(r, A, r0, c1, f1, B))

def profile_modified_hamaus(r, alpha, beta, rs, delta_c, delta_large=0.0, 
        rv=1.0
    ):
    """
    Modified Hamaus et al. (2014) void density profile.

    Extends the Hamaus profile by adding a constant density offset (delta_large)
    to allow for large-scale compensation or uniform overdensity.

    Parameters:
        r (float or array): Radius
        alpha, beta (float): Shape parameters
        rs (float): Void scale radius
        delta_c (float): Central density contrast
        delta_large (float): Large-scale offset (default: 0)
        rv (float): Characteristic void radius (default: 1.0)

    Returns:
        float or array: Density contrast δ(r)
    
    Tests:
        Tested in cosmology_inference_tests/test_profiles.py
        Regression tests: test_integrated_hamaus_regression,
                          test_modified_hamaus_regression
        Unit tests: test_modified_hamaus_output_shape,
                    test_integrated_profile_finite,
                    test_rho_real_matches_hamaus,
                    test_integrated_matches_numerical
    """
    return ((delta_c - delta_large) * (1.0 - (r / rs)**alpha) /
            (1.0 + (r / rv)**beta)) + delta_large

def profile_modified_hamaus_derivative(r,order,alpha,beta,rs,delta_c,
                                       delta_large=0,rv=1.0):
    """
    Returns derivatives of profile_modified_hamaus
    
    Parameters:
        r (float or array): Radius
        order (int): order of the derivative to return
        Other parameters as in profile_modified_hamaus
    
    Returns:
        float or array: Derivative at r
    
    Tests:
        Tested in cosmology_inference_tests/test_profiles.py
        Regression tests: test_profile_modified_hamaus_derivative
    """
    if order not in [0,1,2]:
        raise Exception("Derivative order not valid or not implemented.")
    if order == 0:
        return profile_modified_hamaus(r,alpha,beta,rs,delta_c,
                                       delta_large=delta_large,rv=rv)
    numerator = 1 - (r/rs)**alpha
    denominator = 1 + (r/rv)**beta
    extra_factor = ( (alpha/rs)*(r/rs)**(alpha-1)/numerator
                    +(beta/rv)*(r/rv)**(beta-1)/denominator)
    delta = profile_modified_hamaus(
        r,alpha,beta,rs,delta_c,delta_large=delta_large,rv=rv
    )
    if order == 1:
        return -(delta - delta_large)*extra_factor
    if order == 2:
        deltap = profile_modified_hamaus_derivative(r,1,alpha,beta,rs,delta_c,
                                                    delta_large=delta_large,
                                                    rv=rv)
        return -deltap*extra_factor - (delta - delta_large)*(
            (alpha*(alpha-1)/rs**2)*(r/rs)**(alpha-2)/numerator
            + (beta*(beta-1)/rv**2)*(r/rv)**(beta-2)/denominator
            + (alpha/rs)**2*(r/rs)**(2*alpha - 2)/numerator**2
            - (beta/rv)**2*(r/rv)**(2*beta-2)/denominator**2)



def integrated_profile_modified_hamaus(r, alpha, beta, rs, delta_c,
                                       delta_large=0.0, rv=1.0):
    """
    Integrated (cumulative) version of the modified Hamaus profile.

    Computes the average density contrast Δ(r), i.e., the mass enclosed
    within a sphere of radius r divided by the volume.

    Uses hypergeometric functions to perform analytic integration of δ(r).

    Parameters:
        r (float or array): Radius
        alpha, beta (float): Shape parameters
        rs (float): Void scale radius
        delta_c (float): Central density contrast
        delta_large (float): Large-scale offset (default: 0)
        rv (float): Characteristic void radius (default: 1.0)

    Returns:
        float or array: Cumulative density contrast Δ(r)
    
    Tests:
        Tested in cosmology_inference_tests/test_profiles.py
        Regression tests: test_integrated_hamaus_regression
        Unit tests: test_integrated_profile_finite,
                    test_integrated_matches_numerical,
    """
    arg = (r / rv)**beta
    hyp_1 = scipy.special.hyp2f1(1,3 / beta, 1 + (3 / beta), -arg)
    hyp_2 = scipy.special.hyp2f1(
        1,(alpha + 3) / beta, 1 + ( (alpha + 3) / beta), -arg
    )
    return delta_large + (delta_c - delta_large)*( hyp_1 - 
        ( 3 / (alpha + 3) ) * ( r / rs )**alpha * hyp_2 )



def rho_real(r, *profile_args):
    """
    Wrapper function for the real-space density profile.

    Currently hardcoded to use the modified Hamaus profile. Could be 
    generalized to allow switching between profile models.

    Parameters:
        r (float or array): Radius
        *profile_args: Arguments passed to profile_modified_hamaus

    Returns:
        float or array: Real-space density profile ρ(r)
    
    Tests:
        Tested in cosmology_inference_tests/test_profiles.py
        Regression tests: test_rho_real
        Unit tests: test_rho_real_matches_hamaus
    """
    return profile_modified_hamaus(r, *profile_args)


def get_weights(los_zspace, void_radii, additional_weights=None):
    """
    Compute per-void stacking weights for redshift-space LOS data.

    This combines:
      - 1 / R_eff^3 volume-based scaling (via get_weights_for_stack)
      - Optional user-supplied weights (e.g. inverse variance, reproducibility)

    Parameters:
        los_zspace (list): Per-snapshot list of LOS arrays (per void)
        void_radii (list): Per-snapshot list of void radii
        additional_weights (list or None): Optional weights per void 
                                           (same structure)

    Returns:
        Array of weights for all particles in all voids in all snapshots
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_weights
        Unit tests: test_get_weights_basic
    """
    # Mask voids that actually contribute LOS data
    voids_used = [np.array([len(x) for x in los]) > 0 for los in los_zspace]
    # Remove empty entries
    los_pos = [
        [los[i] for i in np.where(ind)[0]] for los, ind in zip(
            los_zspace, voids_used
        )
    ]
    # Prepare additional weights
    if additional_weights is None:
        weights_list = None
    else:
        all_additional_weights = np.hstack([
            weights[used] for weights, used in zip(
                additional_weights, voids_used
            )
        ])
        weights_list = [
            weights[used] / np.sum(all_additional_weights)
            for weights, used in zip(additional_weights, voids_used)
        ]
    # Volume × optional weighting
    v_weight = get_weights_for_stack(
        los_pos,
        [radii[used] for radii, used in zip(void_radii, voids_used)],
        additional_weights=weights_list
    )
    return v_weight


def get_halo_indices(catalogue):
    """
    Map a void catalogue to halo indices in simulation snapshots.

    Only needed if index mapping isn't already handled by the catalogue class.

    NOTE: Deprecated in favor of direct access through catalogue object.

    Parameters:
        catalogue: Void catalogue with finalCatFrac + indexListShort

    Returns:
        list of arrays: Per-snapshot list of halo indices
    
    Tests:
        Tested in test_stacking_function.py
        Regression tests: test_get_halo_indices
    """
    final_cat = catalogue.get_final_catalogue(void_filter=True)
    halo_indices = [
        -np.ones(len(final_cat), dtype=int) for _ in range(catalogue.numCats)
    ]
    for ns in range(catalogue.numCats):
        have_void = final_cat[:, ns] >= 0
        halo_indices[ns][have_void] = catalogue.indexListShort[ns][
            final_cat[have_void, ns] - 1
        ]
    return halo_indices


#void_radii = catalogue.getMeanProperty("radii",void_filter=True)[0]
#rep_scores = catalogue.property_with_filter(
#    catalogue.finalCatFrac,void_filter=True)

def trim_los_list(los_list, spar_bins, sperp_bins, all_radii):
    """
    Remove voids that have no LOS particles within the stacking volume.

    This filters the LOS list using the (s_par, s_perp) bins and void radii,
    and returns:
      - Trimmed list of LOS particles per void
      - Boolean flags for which voids were retained

    Parameters:
        los_list (list): Per-snapshot list of LOS particle arrays (per void)
        spar_bins (array): LOS bin edges
        sperp_bins (array): Transverse bin edges
        all_radii (list): Per-snapshot list of void radii

    Returns:
        tuple:
            - los_list_trimmed (list of lists): Filtered LOS particle arrays
            - voids_used (list of bool arrays): Mask of retained voids
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions
        Regression tests: test_trim_los_list_regression
        Unit tests: test_trim_los_list_shapes,
                    test_trim_los_list_basic
    """
    los_list_trimmed = get_2d_void_stack_from_los_pos(
        los_list, spar_bins, sperp_bins,
        [all_radii[ns] for ns in range(len(los_list))],
        stacked=False
    )
    voids_used = [np.array([len(x) > 0 for x in los]) for los in los_list]
    return los_list_trimmed, voids_used

def get_trimmed_los_list_per_void(
        los_pos, spar_bins, sperp_bins, void_radii_list
    ):
    """
    Get a flat list of trimmed LOS particle arrays from all voids.

    This wraps `trim_los_list()` and flattens the result to a single
    per-void list, across all snapshots.

    Parameters:
        los_pos (list): Per-snapshot list of LOS arrays (per void)
        spar_bins (array): LOS bin edges
        sperp_bins (array): Transverse bin edges
        void_radii_list (list): Per-snapshot list of void radii

    Returns:
        list: Flattened list of LOS particle arrays per void (trimmed)
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions
        Regression tests: test_get_trimmed_los_list_per_void_regression
        Unit tests: test_get_trimmed_los_list_per_void_shape,
                    test_get_trimmed_los_list_per_void_basic
    """
    los_list_trimmed, _ = trim_los_list(
        los_pos, spar_bins, sperp_bins, void_radii_list
    )
    return sum(los_list_trimmed, [])


def get_lcdm_void_catalogue(snaps, delta_interval=None, dist_max=135,
                            radii_range=[10, 20], centres_file=None,
                            nRandCentres=10000, seed=1000, flattened=True,
                            recompute_centres = False):
    """
    Construct a void selection mask from a ΛCDM simulation by:
        1. Selecting random underdense regions with matching density
        2. Removing overlapping regions
        3. Filtering voids that fall in those regions and meet radius cuts

    Parameters:
        snaps (SnapHandler): Snapshot object with 'void_centres' and 
                             'void_radii'
        delta_interval (tuple or None): Density contrast bounds to select 
                                        regions
        dist_max (float): Radius of the spherical region used
        radii_range (list): Acceptable radius range for void selection
        centres_file (str or None): Pickle file path for caching random regions
        nRandCentres (int): Number of random regions to generate if no cache
        seed (int): Random seed for reproducibility
        flattened (bool): Whether to flatten per-region void mask into a single 
                          list
        recompute_centres (bool): If True, recompute centres from scratch.

    Returns:
        list of arrays: Boolean masks per snapshot indicating selected voids
    
    Tests:
        Tested in test_stacking_functions.py
        Regressio tests: test_get_lcdm_void_catalogue
    """
    boxsize = snaps.boxsize
    # Load or generate random region centres and densities
    if centres_file is not None:
        rand_centres, rand_densities = tools.loadOrRecompute(
            centres_file,
            simulation_tools.get_random_centres_and_densities,
            dist_max,
            snaps["snaps"],
            seed=seed,
            nRandCentres=nRandCentres,
            _recomputeData = recompute_centres
        )
    else:
        rand_centres, rand_densities = \
            simulation_tools.get_random_centres_and_densities(
                dist_max,snaps["snaps"],seed=seed,nRandCentres=nRandCentres
            )
    # Step 1: Filter regions by density (if delta bounds specified)
    region_masks, centres_to_use = simulation_tools.filter_regions_by_density(
        rand_centres, rand_densities, delta_interval
    )
    # Step 2: Prune overlapping regions
    nonoverlapping_indices = simulation_tools.getNonOverlappingCentres(
        centres_to_use, 2 * dist_max, boxsize, returnIndices=True
    )
    selected_region_centres = [
        centres[idx] for centres, idx in zip(
            centres_to_use, nonoverlapping_indices
        )
    ]
    selected_region_masks = [
        mask[idx] for mask, idx in zip(region_masks, nonoverlapping_indices)
    ]
    # Step 3: Compute distances from each void to selected regions
    region_void_dists = simulation_tools.compute_void_distances(
        snaps["void_centres"], selected_region_centres, boxsize
    )
    # Step 4: Apply radius and distance filters
    void_masks_by_region = simulation_tools.filter_voids_by_distance_and_radius(
        region_void_dists, snaps["void_radii"], dist_max, radii_range
    )
    # Step 5: Flatten masks if requested
    if flattened:
        return [simulation_tools.flatten_filter_list(masks)
                for masks in void_masks_by_region]
    else:
        return void_masks_by_region

def get_stacked_void_density_field(snaps, void_radii_lists, void_centre_lists,
                                   spar_bins, sperp_bins, halo_indices=None,
                                   filter_list=None, additional_weights=None,
                                   dist_max=3, rmin=10, rmax=20,
                                   recompute=False, zspace=True,
                                   recompute_zspace=False,
                                   suffix=".lospos_all_zspace2.p",
                                   los_pos=None, **kwargs):
    """
    Compute the 2D stacked density field from LOS particle data for a set of 
    voids.

    Handles filtering, redshift-space conversion, void trimming, and weighted 
    stacking.

    Parameters:
        snaps (SnapHandler): Simulation snapshot handler (must include boxsize, 
                             snaps, etc.)
        void_radii_lists (list): Per-snapshot list of void radii
        void_centre_lists (list): Per-snapshot list of void centers
        spar_bins, sperp_bins (array): Bin edges for LOS and transverse 
                                       directions
        halo_indices (list or None): Indices for each void (optional)
        filter_list (list or None): Optional masks for void selection
        additional_weights (list or None): Optional void weights
        dist_max, (float): Stacking distance threshold
        rmin, rmax, (float) minimum and maximum void radius to consider
        recompute, zspace, recompute_zspace (bool): Control redshift-space 
                                                    position cache
        recompute, (bool): if true, recompute LOS positions in the cache
        zspace, (bool): if true, use redshift space positions, not real space
        recompute_zspace, (bool): if true, recompute redshift space positions.
        suffix (str): File suffix for LOS cache
        los_pos (list or None): Precomputed LOS arrays (optional)
        **kwargs: Passed to internal helper functions

    Returns:
        ndarray: 2D stacked density field
    
    Tests:
        Tested in test_stacking_function.py
        Regression tests: test_get_stacked_void_density_field
    """
    boxsize = snaps.boxsize
    nbar = len(snaps["snaps"][0]) / boxsize**3
    if halo_indices is not None and filter_list is None:
        filter_list = [halo_indices[ns] >= 0 for ns in range(snaps.N)]
    # Get LOS particle data
    if los_pos is None:
        los_pos = get_los_positions_for_all_catalogues(
            snaps["snaps"], snaps["snaps_reverse"],
            void_centre_lists, void_radii_lists,
            all_particles=True,
            void_indices=halo_indices,
            filter_list=filter_list,
            dist_max=dist_max, rmin=rmin, rmax=rmax,
            recompute=recompute,
            zspace=zspace,
            recompute_zspace=recompute_zspace,
            suffix=suffix
        )
    # Trim voids with no useful particles
    los_list_trimmed, voids_used = trim_los_list(
        los_pos, spar_bins, sperp_bins, void_radii_lists
    )
    los_list_per_void = sum(los_list_trimmed, [])
    num_voids = sum(np.sum(u) for u in voids_used)
    # Per-void radius and weights
    void_radii_per_void = np.hstack(
        [r[used] for r, used in zip(void_radii_lists, voids_used)]
    )
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([
            weights[used] for weights, used in zip(
                additional_weights, voids_used
            )
        ])
    else:
        additional_weights_per_void = np.ones(len(void_radii_per_void))
    # Compute stacked field
    return get_2d_field_from_stacked_voids(
        los_list_per_void, sperp_bins, spar_bins,
        void_radii_per_void,
        weights=additional_weights_per_void,
        nbar=nbar
    )



def get_1d_real_space_field(snaps, rbins=None, filter_list=None,
                            additional_weights=None, n_boot=10000, seed=42,
                            halo_indices=None, use_precomputed_profiles=True):
    """
    Compute a bootstrapped real-space 1D void density profile (radial stacking).

    Parameters:
        snaps (SnapHandler): Object with snapshot data and precomputed pair 
                             counts
        rbins (array or None): Radial bin edges. If None, defaults to 
                               linspace(0, 3)
        filter_list (list or None): Per-snapshot list of boolean masks for void 
                                    selection
        additional_weights (list or None): Optional void-level weights
        n_boot (int): Number of bootstrap samples
        seed (int): RNG seed for bootstrap
        halo_indices (list or None): Void indices to use (alternative to 
                                     filter_list)
        use_precomputed_profiles (bool): If True, use stored pair counts

    Returns:
        tuple:
            - rho_mean (array): Bootstrapped mean profile
            - rho_std (array): Profile standard deviation

    Tests:
        Tested in test_stacking_function.yp
        Regression tests: test_get_1d_real_space_field
    """
    boxsize = snaps.boxsize
    nbar = len(snaps["snaps"][0]) / boxsize**3
    # Fallback filter setup
    if halo_indices is not None and filter_list is None:
        filter_list = [halo_indices[ns] >= 0 for ns in range(snaps.N)]
    if filter_list is None:
        filter_list = [
            np.ones(len(x), dtype=bool) for x in snaps["pair_counts"]
        ]
    # Load pair counts
    if use_precomputed_profiles:
        rbins = snaps["radius_bins"][0]
        all_counts = snaps["pair_counts"]
        all_volumes = snaps["bin_volumes"]
    else:
        if rbins is None:
            rbins = np.linspace(0, 3, 31)
        all_counts, all_volumes = [], []
        for ns in range(snaps.N):
            tree = scipy.spatial.cKDTree(
                snaps["snaps"][ns]['pos'], boxsize=boxsize
            )
            counts, volumes = stacking.getPairCounts(
                snaps["void_centres"][ns], snaps["void_radii"][ns],
                snaps["snaps"][ns], rbins,
                nThreads=-1, tree=tree, method="poisson",
                vorVolumes=snaps["cell_volumes"][ns]
            )
            all_counts.append(counts)
            all_volumes.append(volumes)
    # Select voids
    if halo_indices is not None:
        all_antihalos = [
            halo_indices[ns][halo_indices[ns] >= 0] - 1 
            for ns in range(snaps.N)
        ]
    elif filter_list is not None:
        all_antihalos = [np.where(filt)[0] for filt in filter_list]
    else:
        all_antihalos = [np.arange(len(x)) for x in snaps["pair_counts"]]
    # Compute all individual density profiles
    all_profiles = [counts[idx] / (vols[idx] * nbar)
                    for counts, vols, idx in zip(
                        all_counts, all_volumes, all_antihalos
                        )
                    ]
    density = np.vstack(all_profiles)
    # Weights
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([
            weights[used] for weights, used in zip(
                additional_weights, filter_list
            )
        ])
    else:
        additional_weights_per_void = np.ones(density.shape[0])
    # Bootstrap over voids
    np.random.seed(seed)
    num_voids = len(additional_weights_per_void)
    bootstrap_samples = np.random.choice(num_voids, size=(num_voids, n_boot))
    bootstrap_profiles = np.array([
        np.average(density[bootstrap_samples[:, k], :], axis=0,
                   weights=additional_weights_per_void[bootstrap_samples[:, k]])
        for k in tools.progressbar(range(n_boot))
    ]).T
    rho_mean = np.mean(bootstrap_profiles, axis=1)
    rho_std = np.std(bootstrap_profiles, axis=1)
    return rho_mean, rho_std

def get_additional_weights_borg(cat, voids_used=None):
    """
    Get reproducibility-based weights for BORG voids.

    Uses the finalCatFrac property as a proxy for the confidence or 
    reproducibility of a void detection. These are normalized across all voids 
    used in stacking.

    Parameters:
        cat (BorgVoidCatalogue): Void catalogue object
        voids_used (list of bool arrays or None): Optional mask per snapshot

    Returns:
        list of arrays: Normalized reproducibility weights for each void 
                        (per snapshot)
    
    Tests:
        Tested in test_stacking_functions.py
        Regression tests: test_get_additional_weights_borg
    """
    rep_scores = cat.property_with_filter(cat.finalCatFrac, void_filter=True)
    if voids_used is None:
        voids_used = [np.ones(rep_scores.shape[0], dtype=bool)
                      for _ in range(cat.numCats)]
    all_rep_scores = np.hstack([rep_scores[used] for used in voids_used])
    norm_factors = np.sum(all_rep_scores)
    return [rep_scores[used] / norm_factors for used in voids_used]

#-------------------------------------------------------------------------------
# VELOCITY MODELLING

def get_ur_profile_for_void(void_centre,void_radius,rbins,snap,tree=None,
        cell_vols=None,relative_velocity=True,n_threads=-1,
    ):
    """
    Compute the radial velocity profile around a given void
        
    Parameters:
        void_centre (array): Centre of the void
        void_radius (float): Radius of the void
        rbins (array of floats): Radial bins to estimate velocities, in units
                                 of void radius
        snap (pynbody snapshot): Simulation snapshot containing the void
        tree (scipy.cKDTree): KD tree used for speeding up particle finding
                              in the simulation snapshot. Computed from scratch
                              if not provided, but more efficient to precompute
                              this and supply it as an argument if computing
                              profiles for multiple voids.
        cell_vols (array of floats): Volumes of the Voronoi cells around each
                              particle in the snapshot. Used to perform volume
                              weigting of the velocities.
        relative_velocity (bool): If True (default), computed velocity relative
                                  to the centre of the void, taking into 
                                  account the bulk void motion. Otherwise, 
                                  just uses the simulation velocity.
        n_threads (int): Number of threads to use for computing profile. If
                         -1 (default), then use all available threads.

    Returns:
        ur_profile (num_bins component array): velocity profile in the provided 
                                               bins
        Delta_r (num_bins component array): Number density of particles in the 
                                            same bins
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions
        Regression tests: test_get_ur_profile_for_void
    """
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    nbar = len(snap)/boxsize**3
    # KD Tree:
    wrapped = np.all(snap['pos'] > 0) and np.all(snap['pos'] < boxsize)
    wrapped_pos = snap['pos'] if wrapped else snapedit.wrap(
        snap['pos'],boxsize
    )
    wrapped_centre = void_centre if wrapped else snapedit.wrap(
        void_centre,boxsize
    )
    if tree is None:
        tree = scipy.spatial.cKDTree(wrapped_pos,boxsize=boxsize)
    # Get particles around the void:
    indices_list = np.array(tree.query_ball_point(wrapped_centre,\
            rbins[-1]*void_radius,workers=n_threads),dtype=np.int64)
    # Displacement from void centre:
    disp = snapedit.unwrap(snap['pos'][indices_list,:] - void_centre,boxsize)
    r = np.array(np.sqrt(np.sum(disp**2,1)))
    # Velocities of void centre:
    velocities_v = snap['vel'][indices_list,:]
    if cell_vols is None:
        cell_vols = np.ones(len(snap))
    weights = cell_vols[indices_list]
    v_centre = np.average(velocities_v,axis=0,weights=weights)
    # Velocity relative to void centre:
    u = velocities_v - v_centre if relative_velocity else velocities_v
    # Ratio between radial velocity and radius:
    ur_norm = np.sum(np.array(u * disp),1)/r**2
    [indices, counts] = plot.binValues(r,rbins*void_radius)
    Delta_r = np.cumsum(counts)/(4*np.pi*nbar*rbins[1:]**3*void_radius**3/3)-1.0
    # Profile as a function of r:
    ur_profile = np.array([
        np.average(
            ur_norm[ind],weights=cell_vols[indices_list][ind]
        ) if len(ind) > 0 else 0.0
        for ind in indices
    ])
    return ur_profile, Delta_r

def get_all_ur_profiles(centres, radii,rbins,snap,tree=None,cell_vols=None,
                        relative_velocity=True):
    """
    Compute the radial velocity profiles for a stack of voids
    
    Parameters:
        centres (N x 3 array of floats): centres of all voids in the stack
        radii (N component array of float): radii for all voids in the stack
        
        Other parameters as in get_ur_profile_for_void
    
    Returns:
        ur_profiles (N x num_bins array of floats): Velocity profile for each 
                                                    void
        Delta_r_profiles (N x num_bins array of floats): Number density for 
                                                         each void.
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_all_ur_profiles
    """
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    wrapped = np.all(snap['pos'] > 0) and np.all(snap['pos'] < boxsize)
    wrapped_pos = snap['pos'] if wrapped else snapedit.wrap(
        snap['pos'],boxsize
    )
    if tree is None:
        tree = scipy.spatial.cKDTree(wrapped_pos,boxsize=boxsize)
    profiles = [
        get_ur_profile_for_void(void_centre,void_radius,rbins,snap,
                                tree=tree,cell_vols=cell_vols,
                                relative_velocity=True
        )
        for void_centre, void_radius, i in zip(
            centres,radii,tools.progressbar(range(len(centres)))
        )
    ]
    ur_profiles = np.vstack([prof[0] for prof in profiles])
    Delta_r_profiles = np.vstack([prof[1] for prof in profiles])
    return ur_profiles,Delta_r_profiles

def semi_analytic_model(u,alphas,z=0,Om=0.3111,f1=None,h=1,nf1 = 5/9,**kwargs):
    """
    Semi-analytic model for velocities around a void. Uses 1LPT, but then has 
    arbitrary coefficients for higher orders, which are matched to the velocity
    data.
    
    The model is defined as:
    
    v = a*H(a)*f1*(u + \sum_{n=2}^{N}\alpha_n u^n)
    
    were a is the scale factor, H(a) the Hubble rate at that scale factor,
    f1 the linear growth rate, and u is the displacement field for which we 
    want to computet he corresponding velocity. The parameters to be fixed
    are alpha_n from n=2 to n=N.
    
    Parameters:
        u (float or array): Displacement field divided by radius (dimensionless)
        alphas (list or array): Parameters for the model, representing
                                alpha_2 ...alpha_N. Number of parameters is 
                                inferred from alphas, N = len(alphas) + 1
                                (note, N is one more than the number of 
                                parameters).
        z (float): Redshift at which to compute the model.
        Om (float): Matter density parameters
        f1 (float or None): Linear growth rate. If None, the Lambda-CDM value
                            is assumed, calculated at the given redshift from 
                            Om and z.
        h (float): Dimensionless Hubble rate. h = 1 (default) means that
                    distances are assumed to be in units of Mpc/h.
        nf1 (float): Exponent of Omega_m(z) used to approximate the Lambda-CDM
                     value of f1, if f1 is not given.
    
    Returns:
        float or array: Velocities for the given displacement field, divided by
                        radius from the void centre. Units (km/s)*h/Mpc if
                        h = 1, otherwise (km/s)/Mpc.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py
        Unit tests: test_semi_analytic_model_functions_consistency
        Regression tests: test_semi_analytic_model
    """
    if f1 is None:
        Omz = Omega_z(z,Om,**kwargs)
        f1 = Omz**nf1
    a = 1/(1+z)
    H = Hz(z,Om,h=h,**kwargs)
    N = len(alphas) + 1
    sum_term = np.sum([alphas[n-2]*u**n for n in range(2,N+1)],0)
    return a*H*f1*(u + sum_term)

#-------------------------------------------------------------------------------
# SPHERICAL OVERDENSITY MODEL

def Delta_theta(theta,taylor_expand=False,exact = False):
    """
    Computes the cumulative density contrast for a given void development angle,
    assuming the spherical shell model.
    
    Parameters:
        theta (float or array): Development angle
        taylor_expand (bool): If true, force Taylor expansion around theta = 0
                              to avoid problems near theta = 0.
        exact (bool): If true, force the theta > 0 expression.
    
    Returns:
        Delta (float or array): same size as theta. Cumulative density contrast.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py
        Regression tests: test_Delta_theta
        Unit tests: test_theta_of_Delta_basic,
                    test_invert_Delta_theta_scalar_basic,
                    test_get_upper_bound_basic,
                    
    """
    if exact:
        return 9*(np.sinh(theta) - theta)**2/(2*(np.cosh(theta) - 1)**3)
    if taylor_expand:
        return 1 - 3*theta**2/20
    if np.isscalar(theta):
        if theta == np.inf:
            return 0.0
        elif theta < 1e-10:
            return Delta_theta(theta, taylor_expand = True)
        else:
            return Delta_theta(theta, exact=True)
    else:
        small = (theta < 1e-10)
        infinite = (theta == np.inf)
        not_small = np.logical_not(np.logical_or(small,infinite))
        retval = np.zeros(theta.shape)
        retval[small] = Delta_theta(theta[small],taylor_expand=True)
        retval[not_small] = Delta_theta(theta[not_small],exact=True)
        retval[infinite] = 0.0
        return retval

def V_theta(theta,taylor_expand=False,exact = False):
    """
    Computes the radial velocity around voids as a function of the void
    development angle, assuming the spherical shell model.
    
    Parameters:
        theta (float or array): Development angle
        taylor_expand (bool): If true, force Taylor expansion around theta = 0
                              to avoid problems near theta = 0.
        exact (bool): If true, force the theta > 0 expression.
    
    Returns:
        V (float or array): same size as theta. Radial peculiar velocity.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py
        Regression tests: test_V_theta
    """
    if exact:
        return 3*(
            np.sinh(theta)*(np.sinh(theta) - theta)/
            (2*(np.cosh(theta) - 1)**2)
        ) - 1
    if taylor_expand:
        return theta**2/20
    if np.isscalar(theta):
        if theta == np.inf:
            return 0.5
        elif theta < 1e-10:
            return V_theta(theta,taylor_expand=True)
        else:
            return V_theta(theta,exact=True)
    else:
        small = (theta < 1e-10)
        infinite = (theta == np.inf)
        not_small = np.logical_not(np.logical_or(small,infinite))
        retval = np.zeros(theta.shape)
        retval[small] = V_theta(theta[small],taylor_expand=True)
        retval[not_small] = V_theta(theta[not_small],exact=True)
        retval[infinite] = 0.5
        return retval

# Inversion:

def get_upper_bound(Delta,count_max=10):
    """
    Computes a viable upper bound used for finding a solution to the equation
    that gives the development angle. This is used by the root finder to
    bound the solution automatically, so that we can always find a solution.
    
    Parameters:
        Delta (float): Density contrast we are targeting
        count_max (int): Maximum iterations before throwing an error.
    
    Returns:
        theta_max (float): Upper bound on the solution for the development 
                           angle for the supplied Delta.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py
        Regression tests: test_get_upper_bound
        Unit tests: test_get_upper_bound_basic
    """
    if Delta == -1:
        return np.inf
    f = lambda x: Delta_theta(x) - Delta - 1
    theta_max = -np.log(2*(1 + Delta)/9) # Asymtotic guess at solution
    count = 0
    while f(theta_max) > 0 and count < count_max:
        theta_max *= 10
        count += 1
    if count >= count_max:
        raise Exception("Unable to find upper bound on solution.")
    else:
        return theta_max

def invert_Delta_theta_scalar(Delta,theta_min=0,theta_max = None,count_max=10):
    """
    Computes the development angle for a single given cumulative density 
    contrast, essentially inverting Delta_theta(theta).
    
    Parameters:
        Delta (float): Density to find the value of theta for.
        theta_min (float): Lower bound on theta (default 0)
        theta_max (float or None): Upper bound on theta. If not provided, will
                                   be computed using get_upper_bound
        count_max (int): Max iterations used by get_upper_bound
    
    Returns:
        theta (float): Development angle that gives the specified Delta
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py
        Regression tests: test_invert_Delta_theta_scalar
        Unit tests: test_invert_Delta_theta_scalar_basic
    """
    if Delta == -1:
        return np.inf
    f = lambda x: Delta_theta(x) - Delta - 1
    if theta_max is None:
        theta_max = get_upper_bound(Delta,count_max=count_max)
    return scipy.optimize.brentq(f,theta_min,theta_max)

def theta_of_Delta(Delta,count_max = 10):
    """
    Inverse of Delta(theta). Computed for vectors or scalars using 
    invert_Delta_theta_scalar.
    
    Parameters:
        Delta (float or array): Density contrast values for which to find theta.
        count_max (int): Max iterations used by get_upper_bound
    
    Returns:
        theta (float or array): Same size as Delta, the development angle 
                                for each Delta.
    
    Tests:
        Tested in cosmology_inference_tests/test_lpt.py
        Regression tests: test_theta_of_Delta
        Unit tests: test_theta_of_Delta_basic
    """
    theta_min = 0
    # Get bound:
    scalar = np.isscalar(Delta)
    Delta_min = Delta if scalar else np.min(Delta[Delta > -1])
    theta_max = get_upper_bound(Delta_min,count_max=count_max)
    if scalar:
        return invert_Delta_theta_scalar(
            Delta,theta_max=theta_max,count_max=count_max
        )
    else:
        return np.array(
            [
                invert_Delta_theta_scalar(
                    x,theta_max=theta_max,count_max=count_max
                )
                for x in Delta
            ]
        )




#-------------------------------------------------------------------------------
# COVARIANCE MATRIX CALCULATION

def get_void_weights(los_list_trimmed, voids_used, all_radii,
                     additional_weights=None):
    """
    Compute the total stacking weight for each void in the dataset.

    This wraps get_weights_for_stack(), filtering the void radii using the
    `voids_used` boolean masks to match the trimmed LOS list.

    Weights combine:
      - A volume-based scaling (1 / R_eff^3)
      - Optional additional weights (e.g. for S/N or inverse variance)

    NOTE: This function may be deprecated if weights are split into separate
    logical operations (e.g., volume normalization vs S/N weighting).

    Parameters:
        los_list_trimmed (list): List of per-void LOS particle arrays (2D)
        voids_used (list of bool arrays): Per-snapshot mask of valid voids
        all_radii (list of arrays): R_eff for all voids (before masking)
        additional_weights (list or scalar or None): Optional multiplicative 
                                                     weights

    Returns:
        list of arrays: Per-void weights (unstacked)
    
    Tests:
        Tested in cosmology_inference_tests/test_stacking_functions.py
        Regression tests: test_get_void_weights
    """
    # Apply void selection mask to radii
    filtered_radii = [rad[used] for used, rad in zip(voids_used, all_radii)]
    return get_weights_for_stack(
        los_list_trimmed,
        filtered_radii,
        additional_weights=additional_weights,
        stacked=False
    )

def get_covariance_matrix(los_list, void_radii_all,
                          spar_bins, sperp_bins, nbar,
                          additional_weights=None,
                          n_boot=10000, seed=42,
                          lambda_reg=1e-15,
                          cholesky=True, regularise=True,
                          log_field=False, return_mean=False):
    """
    Estimate the covariance matrix (or its Cholesky decomposition) of a
    2D stacked void density field using bootstrap resampling.

    Steps:
        1. Trims voids with no contributing LOS particles
        2. Computes 2D density fields for each individual void
        3. Bootstraps the mean stacked profile over voids
        4. Computes the covariance matrix (or its log)
        5. Optionally regularizes and/or returns Cholesky decomposition

    Parameters:
        los_list (list): Per-snapshot list of LOS particle arrays (per void)
        void_radii_all (list): Per-snapshot list of void radii
        spar_bins (array): LOS bin edges
        sperp_bins (array): Transverse bin edges
        nbar (float): Mean number density
        additional_weights (list or None): Optional void weights per snapshot
        n_boot (int): Number of bootstrap realizations
        seed (int): RNG seed for reproducibility
        lambda_reg (float): Regularization strength for covariance
        cholesky (bool): If True, return lower-triangular Cholesky factor
        regularise (bool): If True, apply regularization to covariance
        log_field (bool): If True, compute covariance of log-density
        return_mean (bool): If True, return the bootstrap mean as well

    Returns:
        cov (ndarray): Covariance matrix (or Cholesky factor if cholesky=True)
        mean (ndarray, optional): Mean field if return_mean=True
    
    Tests:
        Tested in cosmology_inference_tests/test_covariance_and_statistics.py
        Regression tests: test_covariance_matrix_regression
        Unit tests: test_get_covariance_matrix_symmetry,
                    test_get_covariance_matrix_positive_definite
    """
    # --- Step 1: Trim out voids with no contributing particles
    los_list_trimmed, voids_used = trim_los_list(
        los_list, spar_bins, sperp_bins, void_radii_all
    )
    los_per_void = sum(los_list_trimmed, [])
    void_radii_per_void = np.hstack([
        radii[used] for radii, used in zip(void_radii_all, voids_used)
    ])
    num_voids = len(void_radii_per_void)
    # --- Step 2: Compute stacked 2D field per void (flattened)
    stacked_fields = get_2d_fields_per_void(
        los_per_void, sperp_bins, spar_bins,
        void_radii_per_void, nbar=nbar
    ).reshape(num_voids, (len(spar_bins) - 1) * (len(sperp_bins) - 1))
    # --- Step 3: Prepare void weights
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([
            weights[used] for weights, used in zip(
                additional_weights, voids_used
            )
        ])
    else:
        additional_weights_per_void = np.ones(num_voids)
    # --- Step 4: Bootstrap stacking over voids
    np.random.seed(seed)
    bootstrap_samples = np.random.choice(num_voids, size=(num_voids, n_boot))
    bootstrap_stacks = np.array([
        np.average(
            stacked_fields[bootstrap_samples[:, k], :],
            axis=0,
            weights=additional_weights_per_void[bootstrap_samples[:, k]]
        )
        for k in tools.progressbar(range(n_boot))
    ]).T  # shape: [n_bins, n_boot]
    # --- Step 5: Compute covariance matrix
    if log_field:
        log_samples = np.log(bootstrap_stacks)
        finite_mask = np.all(np.isfinite(log_samples), axis=0)
        log_samples = log_samples[:, finite_mask]
        cov = np.cov(log_samples)
        mean = np.mean(log_samples, axis=1)
    else:
        cov = np.cov(bootstrap_stacks)
        mean = np.mean(bootstrap_stacks, axis=1)
    # --- Step 6: Regularization
    if regularise:
        cov = regularise_covariance(cov, lambda_reg=lambda_reg)
    # --- Step 7: Cholesky decomposition (if requested)
    if cholesky:
        cov = scipy.linalg.cholesky(cov, lower=True)
    if return_mean:
        return cov, mean
    else:
        return cov

#-------------------------------------------------------------------------------
# COSMOLOGICAL INFERENCE

def get_mle_estimate(initial_guess, theta_ranges, 
        log_likelihood,*args, **kwargs
    ):
    """
    Compute the Maximum Likelihood Estimate (MLE) for parameters,
    by minimizing the negative log-likelihood function.

    Parameters:
        initial_guess (array): Initial guess for parameters
        theta_ranges (list of tuples): Parameter bounds
        log_likelihood (function): Log likelihood to file mle estimate for.
        
        *args: Arguments passed to the likelihood function
        **kwargs: Keyword args for the likelihood

    Returns:
        OptimizeResult: Output from scipy.optimize.minimize
    
    Tests:
        Tested in test_likelihood_and_posterior.py
        Regression tests: test_get_mle_estimate
    """
    nll = lambda theta: -log_likelihood(theta, *args, **kwargs)
    mle_estimate = scipy.optimize.minimize(
        nll,
        initial_guess,
        bounds=theta_ranges
    )
    return mle_estimate.x

def generate_scoord_grid(sperp_bins, spar_bins):
    """
    Generate (s_par, s_perp) bin-centre coordinate pairs for all bins
    in the 2D stacked void field.

    Parameters:
        sperp_bins (array): Bin edges in transverse direction
        spar_bins (array): Bin edges in line-of-sight direction

    Returns:
        ndarray: Array of shape (N_bins, 2), where each row is [s_par, s_perp]
    
    Tests:
        Tested in test_likelihood_and_posterior.py
        Regression tests: test_generate_scoord_grid
        Unit tests: test_generate_scoord_grid_shape
    """
    spar = np.hstack([
        s * np.ones(len(sperp_bins) - 1)
        for s in plot.binCentres(spar_bins)
    ])
    sperp = np.hstack([
        plot.binCentres(sperp_bins)
        for _ in plot.binCentres(spar_bins)
    ])
    return np.vstack([spar, sperp]).T


def generate_data_filter(cov, mean, scoords, cov_thresh=5, srad_thresh=1.5):
    """
    Apply a data mask to exclude noisy or poorly constrained bins.

    Parameters:
        cov (ndarray): Covariance matrix
        mean (ndarray): Mean stacked profile
        scoords (ndarray): Grid coordinates (s_par, s_perp)
        cov_thresh (float): Inverse S/N threshold (1/sigma > cov_thresh)
        srad_thresh (float): Max radial distance to include (in units of R_eff)

    Returns:
        ndarray: Array of indices for bins that pass the filter
    
    Tests:
        Tested in test_likelihood_and_posterior.py
        Regression tests: test_generate_data_filter
    """
    norm_cov = cov / np.outer(mean, mean)
    inv_sigma = 1.0 / np.sqrt(np.diag(norm_cov))
    radial_dist = np.sqrt(np.sum(scoords**2, axis=1))
    data_filter = np.where(
        (inv_sigma > cov_thresh) & (radial_dist < srad_thresh)
    )[0]
    return data_filter

def log_likelihood_profile(theta, x, y, yerr, profile_model):
    """
    Gaussian log-likelihood for a real-space density profile model.

    Parameters:
        theta (array): Profile parameters
        x (array): Radial coordinates
        y (array): Observed densities
        yerr (array): Uncertainties
        profile_model (function): Model function ρ(x, *theta)

    Returns:
        float: Log-likelihood
    
    Tests:
        Tested in 
        Regression tests: test_log_likelihood_profile
        Unit tests: test_log_likelihood_profile_basic,
                    test_log_likelihood_output,
                    test_log_likelihood_penalizes_wrong_model
        
    """
    model = profile_model(x, *theta)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))


def get_profile_parameters_fixed(ri, rhoi, sigma_rhoi,
        model=profile_modified_hamaus,
        initial=np.array([1.0, 1.0, 1.0, -0.2, 0.0, 1.0]),
        bounds=[(0, None), (0, None), (0, None),(-1, 0), (-1, 1), (0, 2)]
    ):
    """
    Fit a fixed-profile model to a 1D void density profile using MLE.

    Parameters:
        ri (array): Radial bin centres (in units of R_eff)
        rhoi (array): Measured density values at ri
        sigma_rhoi (array): Uncertainties on density
        model (function): Parametric profile function δ(r, *params)
        initial (array): Initial guess for profile parameters
        bounds (list of tuples): Parameter bounds for optimization

    Returns:
        array: Best-fit parameter values
    
    Tests:
        Tested in test_likelihood_and_posterior.py
        Regression tests: test_profile_fit_regression
        Unit tests: test_get_profile_parameters_fixed_convergence,
                    test_get_profile_parameters_fixed_converges
    """
    # Define negative log-likelihood function for optimizer
    nll = lambda *theta: -log_likelihood_profile(*theta)
    # Perform bounded minimization
    sol = scipy.optimize.minimize(
        nll,
        initial,
        bounds=bounds,
        args=(ri, rhoi, sigma_rhoi, model)
    )
    return sol.x



def run_inference(data_field, theta_ranges_list, theta_initial, filename,
                  log_probability, *args,
                  redo_chain=False,
                  backup_start=True,
                  nwalkers=64, sample="all", n_mcmc=10000,
                  disp=1e-4, Om_fid=0.3111, max_n=1000000, z=0.0225,
                  parallel=False, batch_size=100, n_batches=100,
                  data_filter=None, autocorr_file=None, **kwargs):
    """
    Run MCMC inference using emcee to sample posterior over cosmological + 
    profile parameters.

    Parameters:
        data_field (array): Flattened observed density field (after filtering)
        theta_ranges_list (list of tuples): Parameter bounds
        theta_initial (array): Initial guess for parameters
        filename (str): Path to HDF5 file for emcee backend
        log_probability (function): Log-posterior function to sample
        *args: Arguments passed to log_probability
        redo_chain (bool): If True, erase and restart chain
        backup_start (bool): If True, backup existing chain to .old
        nwalkers (int): Number of walkers
        sample (str or array): "all" or boolean array specifying sampled 
                               parameters
        n_mcmc (int): Number of MCMC steps
        disp (float): Walker spread around initial guess
        Om_fid (float): Fiducial Omega_m
        max_n (int): Maximum number of samples to allow
        z (float): Redshift
        parallel (bool): If True, use multiprocessing Pool
        batch_size (int): MCMC steps per batch
        n_batches (int): Max number of batches
        data_filter (array or None): Indices of used bins
        autocorr_file (str or None): Path to save autocorrelation data
        **kwargs: Passed to log_probability

    Returns:
        tau (array): Autocorrelation time estimates for each parameter
        sampler (emcee.EnsembleSampler): Final MCMC sampler object
    
    Tests:
        Tested in test_inference_core.py
        Unit tests: test_run_inference_basic
    """
    if emcee is None:
        raise Exception("emcee not found.")
    if sample == "all":
        sample = np.array([True for _ in theta_ranges_list])
    ndims = np.sum(sample)
    ndata = len(data_field)
    if data_filter is None:
        data_filter = np.arange(ndata)
    # --- Initial walker positions
    initial = theta_initial + disp * np.random.randn(nwalkers, ndims)
    # --- Optional: backup old file
    filename_initial = filename + ".old"
    if backup_start:
        os.system(f"cp {filename} {filename_initial}")
    backend = emcee.backends.HDFBackend(filename)
    if redo_chain:
        backend.reset(nwalkers, ndims)
    # --- Run in parallel
    if parallel:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndims, log_probability_aptest_parallel,
                args=(z,),
                kwargs={'Om_fid': Om_fid, 'cholesky': True,
                        'tabulate_inverse': True, 'sample_epsilon': True},
                backend=backend, pool=pool
            )
            sampler.run_mcmc(initial, n_mcmc, progress=True)
    else:
        # --- Serial run with convergence monitoring
        sampler = emcee.EnsembleSampler(
            nwalkers, ndims, log_probability,
            args=args,
            kwargs=kwargs,
            backend=backend
        )
        # Load or initialize autocorrelation array
        if redo_chain or autocorr_file is None:
            autocorr = np.zeros((ndims, 0))
        else:
            autocorr = np.load(autocorr_file)
        previously_run = False
        old_tau = np.inf
        for k in range(n_batches):
            if (k == 0 and redo_chain) or not previously_run:
                sampler.run_mcmc(initial, batch_size, progress=True)
            else:
                sampler.run_mcmc(None, batch_size, progress=True)
            # --- Autocorrelation monitoring
            try:
                tau = sampler.get_autocorr_time(tol=0)
            except emcee.autocorr.AutocorrError:
                tau = old_tau  # fallback
            autocorr = np.hstack([autocorr, tau.reshape((ndims, 1))])
            if autocorr_file is not None:
                np.save(autocorr_file, autocorr)
            # --- Convergence check
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
            previously_run = True
    return tau, sampler

from void_analysis.tools import get_finite_range

def run_inference_pipeline(field, cov, mean, sperp_bins, spar_bins,
        ri, delta_i, sigma_delta_i,log_field=False,infer_profile_args=True,
        infer_velocity_args = True,
        tabulate_inverse=True,cholesky=True,sample_epsilon=True,
        filter_data=False,z=0.0225,lambda_cut=1e-23,lambda_ref=1e-27,
        profile_param_ranges=[
            [0, np.inf], [0, np.inf], [0, np.inf],[-1, 0], [-1, 1], [0, 2]
        ],vel_profile_param_ranges = [],om_ranges=[[0.1, 0.5]],
        eps_ranges=[[0.9, 1.1]],f_ranges=[[0, 1]],Om_fid=0.3111,
        filename="inference_weighted.h5",autocorr_filename="autocorr.npy",
        disp=1e-2,nwalkers=64,n_mcmc=10000,max_n=1000000,batch_size=100,
        nbatch=100,redo_chain=False,backup_start=True,F_inv = None,
        delta_profile=profile_modified_hamaus,
        Delta_profile=integrated_profile_modified_hamaus,
        vel_model = void_los_velocity_ratio_1lpt,
        dvel_dlogr_model = void_los_velocity_ratio_derivative_1lpt,
        vel_params_guess=None
    ):
    """
    Full inference pipeline to constrain cosmological parameters from
    stacked void density fields in redshift space.

    Steps:
        1. Compute inverse covariance or Cholesky factor
        2. Filter data spatially and based on S/N
        3. Define density profile model (fixed or inferred)
        4. Precompute redshift-space inverse mapping (optional)
        5. Prepare parameter bounds and priors
        6. Run MCMC inference

    Parameters:
        field (2D array): Stacked void field
        cov (2D array): Covariance matrix for the field
        sperp_bins, spar_bins (arrays): Bin edges in LOS and transverse 
                                        directions
        ri, delta_i, sigma_delta_i (arrays): Real-space profile and 
                                             uncertainties
        log_field (bool): If True, use log-density field
        infer_profile_args (bool): If True, sample profile parameters
        infer_velocity_args (bool): If True, sample velocity profile parameters
        tabulate_inverse (bool): If True, precompute redshift-space inverse 
                                 mapping
        cholesky (bool): If True, use Cholesky factor of covariance
        sample_epsilon (bool): If True, sample epsilon and f instead of Om and f
        filter_data (bool): If True, apply data mask to reduce noise
        z (float): Redshift
        lambda_cut (float): Eigenvalue cutoff for filtering singular modes
        lambda_ref (float): Tikhonov regularization for nonsingular subspace
        profile_param_ranges (list): Parameter bounds for profile model
        om_ranges, eps_ranges, f_ranges (list): Bounds for Om/epsilon, f
        Om_fid (float): Fiducial Omega_m for model evaluation
        filename (str): HDF5 file to save MCMC chain
        autocorr_filename (str): Path to store autocorrelation history
        disp (float): Initial walker spread
        nwalkers (int): Number of MCMC walkers
        n_mcmc (int): Total MCMC steps
        max_n (int): Max samples (unused)
        batch_size (int): Steps per convergence-check batch
        nbatch (int): Max number of batches
        redo_chain (bool): If True, overwrite previous chain
        backup_start (bool): Backup old chain before overwrite
        F_inv (function): Tabulated inverse function for mapping between real
                          and redshift space. Will be computed if not supplied.
        delta_profile, Delta_profile (functions): Density and cumulative 
                                                  profile models

    Returns:
        tau (array): Autocorrelation times
        sampler (EnsembleSampler): Final MCMC sampler object
    
    Tests:
        Tested in test_inference_core.py
        Unit tests: test_run_inference_pipeline_basic
    """
    # --- Step 1: Covariance preparation
    if cholesky:
        inverse_matrix = scipy.linalg.cholesky(cov, lower=True)
    else:
        inverse_matrix = get_inverse_covariance(cov, lambda_reg=lambda_ref)
    # --- Step 2: Filter data
    scoords = generate_scoord_grid(sperp_bins, spar_bins)
    if filter_data:
        data_filter = generate_data_filter(cov, mean, scoords)
    else:
        data_filter = np.ones(field.flatten().shape, dtype=bool)
    data_field = field.flatten()[data_filter]
    # --- Step 3: Choose profile model
    if infer_profile_args:
        delta_func = delta_profile
        Delta_func = Delta_profile
        rho_real = lambda *args: delta_profile(*args) + 1.0
    else:
        profile_params = get_profile_parameters_fixed(
            ri, delta_i, sigma_delta_i, model=delta_profile
        )
        delta_func = lambda r: delta_profile(r, *profile_params)
        Delta_func = lambda r: Delta_profile(r, *profile_params)
        rho_real = lambda r: delta_func(r) + 1.0
    # --- Step 4: Parameter bounds and initial guess
    if sample_epsilon:
        initial_guess_MG = np.array([1.0, f_lcdm(z, Om_fid)])
        theta_ranges = (
            eps_ranges + f_ranges + profile_param_ranges
             + vel_profile_param_ranges
        )
    else:
        initial_guess_MG = np.array([Om_fid, f_lcdm(z, Om_fid)])
        theta_ranges = (
            om_ranges + f_ranges + profile_param_ranges
             + vel_profile_param_ranges
        )
    initial_guess = initial_guess_MG
    if infer_profile_args:
        profile_params = get_profile_parameters_fixed(
            ri, delta_i, sigma_delta_i
        )
        initial_guess = np.hstack([initial_guess, profile_params])
    if infer_velocity_args:
        if vel_params_guess is None:
            regular_ranges = [
                get_finite_range(ran) for ran in vel_profile_param_ranges
            ]
            vel_params_guess = np.array([np.mean(x) for x in regular_ranges])
        initial_guess = np.hstack([initial_guess, vel_params_guess])
    # --- Step 5: Filter singular modes
    Umap, good_eig = get_nonsingular_subspace(
        cov, lambda_reg=lambda_ref,
        lambda_cut=lambda_cut, normalised_cov=False,
        mu=mean)
    # --- Step 6: Assemble args and kwargs for MCMC
    args = (
        data_field,
        scoords[data_filter, :],
        inverse_matrix[data_filter][:, data_filter],
        z,
        Delta_func,
        delta_func,
        rho_real
    )
    kwargs = {
        'cholesky': cholesky,
        'tabulate_inverse': tabulate_inverse,
        'sample_epsilon': sample_epsilon,
        'theta_ranges': theta_ranges,
        'singular': False,
        'Umap': Umap,
        'good_eig': good_eig,
        'F_inv': F_inv,
        'log_density': log_field,
        'infer_profile_args':infer_profile_args,
        'vel_model':vel_model,
        'dvel_dlogr_model':dvel_dlogr_model
    }
    # --- Step 8: Run MCMC
    tau, sampler = run_inference(
        data_field, theta_ranges, initial_guess, filename,
        log_probability_aptest, *args,redo_chain=redo_chain,
        backup_start=backup_start,nwalkers=nwalkers,sample="all",n_mcmc=n_mcmc,
        disp=disp,max_n=max_n,z=z,parallel=False,Om_fid=Om_fid,
        batch_size=batch_size,n_batches=nbatch,autocorr_file=autocorr_filename,
        **kwargs
    )
    return tau, sampler

#-------------------------------------------------------------------------------
# PLOTS

def deriv_factor(p):
    return p**2 + p/3 - Fraction(2,3)

def plot_all_Deltaf(uvals = None,
                    labels=None,
                    colors=None,
                    styles = ["-","--","-","--","-"],
                    xlabel= "$u = S_{1r}/r$",
                    ylabel="$\Delta_f$",
                    ylim = [-1.1,1.1],
                    xlim = [-1,1],
                    leg_loc="upper right",
                    adjustments = {"left":0.25,"right":0.95,
                                   "bottom":0.15,"top":0.95},
                    savename=None):
    if uvals is None:
        uvals = np.linspace(-1,1,1000)
    # Work out all the coefficients using fractions package, to avoid errors:
    D1eds = Fraction(1,1)
    Gfactor = Fraction(2,3)
    power1 = Fraction(2,3)
    D2eds = -Gfactor*D1eds**2/deriv_factor(2*power1)
    D3aeds = -2*Gfactor*(D1eds**3)/deriv_factor(3*power1)
    D3beds = -2*Gfactor*(D1eds**3*(D2eds/D1eds**2 - 1))/deriv_factor(3*power1)
    D4aeds = 2*Gfactor*D1eds*(2*D1eds**3 - D3aeds)/deriv_factor(4*power1)
    D4beds = 2*Gfactor*D1eds*(2*D1eds*D2eds - 2*D1eds**3
                               - D3beds)/deriv_factor(4*power1)
    D4ceds = Gfactor*(2*D1eds**2*D2eds - D2eds**2)/deriv_factor(4*power1)
    D4deds = Gfactor*D1eds**4/deriv_factor(4*power1)
    D5aeds = -2*Gfactor*D1eds*(4*D1eds**4 - 4*D1eds*D3aeds
                                + D4aeds)/deriv_factor(5*power1)
    D5beds = -2*Gfactor*D1eds*(4*D1eds**2*D2eds - 4*D1eds**4 - 2*D1eds*D3beds
                                + D4beds)/deriv_factor(5*power1)
    D5ceds = -2*Gfactor*D1eds*(2*D1eds**2*D2eds - D2eds**2
                                + D4ceds)/deriv_factor(5*power1)
    D5deds = -2*Gfactor*D1eds*(D1eds**4 + D4deds)/deriv_factor(5*power1)
    D5eeds = 2*Gfactor*(2*D1eds**3*D2eds - D3aeds*D2eds
                        + D3aeds*D1eds**2)/deriv_factor(5*power1)
    D5feds = 2*Gfactor*(2*D1eds*D2eds**2 - 2*D1eds**3*D2eds - D3beds*D2eds
                        + D3beds*D1eds**2)/deriv_factor(5*power1)
    # Values of psi:
    psir1 = lambda u: D1eds*u
    psir2 = lambda u: D1eds*u + D2eds*u**2
    psir3 = lambda u: psir2(u) + (D3aeds/3 + D3beds)*u**3
    psir4 = lambda u: psir3(u) + (D4aeds/3 + D4beds + D4ceds + D4deds)*u**4
    psir5 = lambda u: psir4(u) + (D5aeds/3 + D5beds + D5ceds + D5deds
                                   + D5eeds/3 + D5feds)*u**5
    Delta_f_func = lambda u :-3*u + 3*u**2 - u**3
    # Plot:
    plt.clf()
    fig, ax = plt.subplots()
    labels = [f"{n}LPT" for n in range(1,6)] if labels is None else labels
    colors = [seabornColormap[n] 
              for n in [1,2,3,4,5]] if colors is None else colors
    psir_list = [psir1,psir2,psir3,psir4,psir5]
    for label, color, style, psir in zip(labels, colors, styles,psir_list):
        plt.plot(
            uvals,Delta_f_func(psir(uvals)),color=color,
            linestyle=style,label=label
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(frameon=False,loc=leg_loc)
    #plt.tight_layout()
    plt.subplots_adjust(**adjustments)
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plot_all_pn(Om = 0.3111,
                uvals=None,
                textwidth=7.1014,
                figsize=None,
                colors=None,
                styles = ["-","--","-","--","-"],
                labels=None,
                xlabel="$u = S_{1r}/r$",
                ylabel="$p_n(u), \Delta_f$",
                ylim=[-1.1,1.1],
                xlim=[-1,1],leg_loc="upper left",
                adjustments = {"left":0.25,"right":0.95,
                               "bottom":0.15,"top":0.95},
                savename=None):
    # Plot of the Initial conditions polynomial:
    # Coefficients of the polynomial:
    D10 = D1(0,Om,normalised=True)
    D2z = lambda Om: -(3/7)*D10**2*Om**(-1/143)
    D3az = lambda Om: -(1/3)*D10**3*Om**(-4/275)
    D3bz = lambda Om: (10/21)*D10**3*Om**(-269/17857)
    A3z = lambda Om: 3*D10*D2z(Om) - D3az(Om) - 3*D3bz(Om) - D10**3
    A2z = lambda Om: 3*(D10**2 - D2z(Om))
    A1z = -3*D10
    p3 = lambda u: A3z(Om)*u**3 + A2z(Om)*u**2 + A1z*u
    p2 = lambda u: A2z(Om)*u**2 + A1z*u
    p1 = lambda u: A1z*u
    # Minimum value of p2:
    p2min = p2(-A1z/(2*A2z(Om)))
    uvals = np.linspace(-1,1,1000) if uvals is None else uvals
    # Plot:
    plt.clf()
    figsize = (0.45*textwidth,0.45*textwidth) if figsize is None else figsize
    colors = [seabornColormap[n] 
              for n in [1,2,3,4,6]] if colors is None else colors
    labels = [f"$p_{n}(u)$" for n in range(1,6)] if labels is None else labels
    pfuncs = [p1,p2,p3,p4,p5]
    fig, ax = plt.subplots(figsize = figsize)
    for pn, color,style, label in zip(pfuncs, colors, styles, labels):
        plt.plot(uvals,pn(uvals),color=color,linestyle=style,label=label)
    plt.axhline(-1.0,color="grey",linestyle=":")
    plt.axhline(p2min,color="grey",linestyle="-.",
                label="$\\Delta_f = $" + ("%.2g" % p2min))
    plt.axhline(0.0,color="grey",linestyle=":")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(frameon=False,loc=leg_loc)
    #plt.tight_layout()
    plt.subplots_adjust(**adjustments)
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plot_pn_inverses(Delta_xrange = None,
                     Om=0.3111,
                     colors = None,
                     labels=None,
                     styles = [":",":",":"],
                     ylim=[0,1],
                     xlim=[-1,0],
                     xlabel='$\\Delta_f$',
                     ylabel='$u = S_{1r}/r$',
                     savename=None):
    # Plot of the inverses:
    Delta_xrange = (np.linspace(-1,0.0,1000) 
        if Delta_xrange is None else Delta_xrange)
    pninvs = [get_initial_condition(Delta_xrange,order=n,Om=Om)
                           for n in range(1,4)]
    colors = [seabornColormap[n] for n in [1,2,3]] if colors is None else colors
    labels = [f"{n}LPT" for n in range(1,4)] if labels is None else labels
    plt.clf()
    for pninv, color, label, style in zip(pninvs, colors, labels, styles):
        plt.plot(Delta_xrange,pninv,color=color,linestyle=style,label=label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plot_inverse_r(rbin_centres,Delta_mean,
                   Om=0.3111,
                   colors=None,
                   labels=None,
                   styles = [":",":",":"],
                   ylim=[0,1],
                   xlim=[0,3],
                   xlabel='$r/r_{\\mathrm{eff}}$',
                   ylabel='$u = S_{1r}/r$',
                   savename=None):
    # Plot of S1r:
    plt.clf()
    pninv_rs = [get_initial_condition(Delta_mean,order=n,Om=Om) 
                for n in range(1,4)]
    colors = [seabornColormap[n] for n in [1,2,3]] if colors is None else colors
    labels = [f"{n}LPT" for n in range(1,4)] if labels is None else labels
    for pninv_r, color, label, style in zip(pninv_rs, colors, labels, styles):
        plt.plot(rbin_centres,pninv_r,color=color,linestyle=style,label=label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    if savename is None:
        plt.savefig(savename)
    plt.show()


def plot_velocity_profiles(
        rbin_centres,
        ur,
        Delta,
        ax=None,
        z_void=0,
        Om_fid=0.3111,
        filename=None,
        ur_range=None,
        ur_ratio=True,
        Delta_r=None,
        ylabel="$u_r/r [h\\mathrm{kms}^{-1}\\mathrm{Mpc}^{-1}]$",
        xlabel="$r/r_{\\mathrm{eff}}$",
        velocity=False,
        treat_2lpt_separately=True,
        show_error_estimates=False,
        ylim=[0,20],
        difference_plot=False,
        colors=None,
        labels=None,
        styles=[":" for _ in range(1,6)],
        **kwargs
    ):
    plt.clf()
    colors = [seabornColormap[n] 
              for n in [1,2,3,4,8]] if colors is None else colors
    labels = [f"{n}LPT Model" for n in range(1,6)] if labels is None else labels
    if ax is None:
        fig, ax = plt.subplots()
    print(kwargs)
    radial_fraction = (not velocity)
    Delta_val = Delta_r if Delta_r is not None else Delta
    Psir_ratio_n = [
        spherical_lpt_displacement(
            1.0,Delta_val,z=z_void,Om=Om_fid,fixed_delta=True,order=n
        )
        for n in range(1,6)
    ]
    # 2LPT is a special case:
    kwargs2 = dict(kwargs)
    kwargs_list = [kwargs,kwargs2,kwargs,kwargs,kwargs]
    if treat_2lpt_separately:
        kwargs2['correct_ics'] = False
    if Delta_r is not None:
        u_pref_nlpt = [
            spherical_lpt_velocity(
                rbin_centres,Delta_r,order=n,Om=Om_fid,
                radial_fraction=radial_fraction,**kw
            )
            for n, kw in zip(range(1,6), kwargs_list)
        ]
    else:
        u_pref_nlpt = [
            spherical_lpt_velocity(
                rbin_centres,Delta,order=n,Om=Om_fid,
                radial_fraction=radial_fraction,**kw
            )
            for n, kw in zip(range(1,6), kwargs_list)
        ]
    if difference_plot:
        for upred in u_pref_nlpt:
            upred -= ur
            upred /= ur
        if ur_range is not None:
            ur_upper = (ur_range[1] - ur)/ur
            ur_lower = (ur_range[0] - ur)/ur
    elif ur_range is not None:
        ur_upper = ur_range[1]
        ur_lower = ur_range[0]
    if ur_ratio:
        if ur_range is not None:
            ax.fill_between(
                rbin_centres,ur_lower,ur_upper,
                color=seabornColormap[0],label="$Simulation, 68\\% interval$",
                alpha=0.5,
            )
        if not difference_plot:
            ax.plot(rbin_centres,ur,
                     color=seabornColormap[0],label="$Simulation$",alpha=0.5,
            )
        for upred, style, label, color in zip(u_pref_nlpt, styles, labels, 
                                              colors):
            ax.plot(rbin_centres,upred,linestyle=style,
                     label=label,color=color,
            )
    else:
        if ur_range is not None:
            ax.fill_between(
                rbin_centres,ur_range[0]*rbin_centres,ur_range[1]*rbin_centres,
                color=seabornColormap[0],label="$Simulation, 68\\% interval$",
                alpha=0.5,
            )
        ax.plot(rbin_centres,ur*rbin_centres,
                 color=seabornColormap[0],label="$Simulation$",alpha=0.5,
        )
        for upred, style, label, color in zip(u_pref_nlpt, styles, labels, 
                                              colors):
            ax.plot(rbin_centres,upred*rbin_centres,linestyle=style,
                     label=label,color=color,
            )
    if show_error_estimates:
        if ur_ratio:
            factor = 1
        else:
            factor = rbin_centres
        error_labels = [f"Expected {n+1}LPT corrections" for n in range(1,6)]
        for upred, psir, n, color, label in zip(u_pred_nlpt,Psir_ratio_n,
                                                range(1,6),colors,error_labels):
            
            ax.fill_between(
                rbin_centres,upred*(1 - psir**n)*factor,
                upred*(1 + psir**n)*factor,alpha=0.5,color=color,label=label
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    plt.legend(frameon=False)
    if filename is not None:
        plt.savefig(filename)
    plt.show()



# Density comparison:
def plot_convergence(
        Delta_mean,
        perturbative_consistency = False,
        Om = 0.3111,
        perturbative_ics=False,
        use_linear_on_fail=False,
        correct_ics = True,
        eulerian_radius=True,
        taylor_expand=False,
        radial_fraction=True,
        z=0,
        textwidth=7.1014,
        figsize=None,
        savename=None
    ):
    if perturbative_consistency:
        Delta_nlpt = [
            get_delta_lpt(
                Delta_mean,Om=Om,order=1,perturbative_ics=perturbative_ics,
                use_linear_on_fail=use_linear_on_fail,correct_ics=correct_ics,
                return_all=True
            )
        ]
        for n in range(2,4):
            Delta_nlpt.append(
                np.sum(
                    get_delta_lpt(
                        Delta_mean,Om=Om,order=n,
                        perturbative_ics=perturbative_ics,
                        use_linear_on_fail=use_linear_on_fail,
                        correct_ics=correct_ics,return_all=True
                    ),0
                )
            )
    else:
        Delta_nlpt = [
            get_delta_lpt(
                Delta_mean,Om=Om,order=n,perturbative_ics=perturbative_ics,
                use_linear_on_fail=use_linear_on_fail,correct_ics=correct_ics
            )
            for n in range(1,4)
        ]
    Psir_ratio_n = [
        spherical_lpt_displacement(
            1.0,Delta_mean,z=z,Om=Om,fixed_delta=True,order=n,
            perturbative_ics=perturbative_ics,
            use_linear_on_fail=use_linear_on_fail,correct_ics=correct_ics
        )
        for n in range(1,6)
    ]
    vr_ratio_n = [
        spherical_lpt_velocity(
            1.0,Delta_mean,z=z,Om=Om,fixed_delta=True,order=n,
            perturbative_ics=perturbative_ics,
            use_linear_on_fail=use_linear_on_fail,
            eulerian_radius=eulerian_radius,taylor_expand=taylor_expand,
            return_all=True,radial_fraction=radial_fraction
        )
        for n in range(1,6)
    ]
    vr_ratio_5 = vr_ratio_n[4]
    Psir_ratio_5 = Psir_ratio_n[5]
    ratios = np.cumsum(Psir_ratio_5,0)/np.sum(Psir_ratio_5,0)
    plt.clf()
    figsize = (0.45*textwidth,0.45*textwidth) if figsize is None else figsize
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(
        rbin_centres,Psir_ratio_5[1]/Psir_ratio_5[2],color=seabornColormap[2],
        label="2nd/3rd (displacement)",linestyle='--'
    )
    plt.plot(
        rbin_centres,Psir_ratio_5[3]/Psir_ratio_5[4],color=seabornColormap[4],
        label="4th/5th (displacement)",linestyle='--'
    )
    plt.plot(
        rbin_centres,vr_ratio_5[1]/vr_ratio_5[2],color=seabornColormap[2],
        label="2nd/3rd (velocity)",linestyle=':'
    )
    plt.plot(
        rbin_centres,vr_ratio_5[3]/vr_ratio_5[4],color=seabornColormap[4],
        label="4th/5th (velocity)",linestyle=':'
    )
    plt.legend(frameon=False,loc="lower right")
    plt.xlabel("$r/r_{\\mathrm{eff}}$")
    plt.ylabel("Ratio compared to next order term.")
    plt.ylim([-5,0])
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plot_psi_true(
        Delta_mean,
        order=5,
        z=0,
        Om = 0.3111,
        perturbative_ics=False,
        use_linear_on_fail=False,
        correct_ics = True,
        savename=None
    ):
    Psi_true = 1/(np.cbrt(1 + Delta_mean)) - 1
    Psir_ratio_N = spherical_lpt_displacement(
        1.0,Delta_mean,z=z,Om=Om,fixed_delta=True,order=order,
        perturbative_ics=perturbative_ics,return_all=True,
        use_linear_on_fail=use_linear_on_fail,correct_ics=correct_ics
    )
    plt.clf()
    colors = [seabornColormap[n] for n in [1,2,3,4,8]]
    labels = [f"n={n}" for n in range(1,6)]
    for k in range(0,order):
        plt.plot(
            rbin_centres, Psir_ratio_N[k]/Psi_true,
            color=colors[k],linestyle="-",
            label = labels[k]
        )
    plt.xlabel("$r/r_{\\mathrm{eff}}$")
    plt.ylabel("$\\Psi^{(n)}/\\Psi_{\\mathrm{exact}}$")
    plt.axhline(1.0,color="grey",linestyle=':')
    plt.legend(frameon=False)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plot_velocity_model(
        rbin_centres,
        Delta_mean,
        ur_range,
        ur_mean,
        alphas_list,
        u_filter=None,
        relative = True,
        z=0,
        Om = 0.3111,
        ylim=[-5,5],
        legend_params={"frameon":False,"ncol":2},
        savename=None
    ):
    plt.clf()
    fig, ax = plt.subplots()
    u_filter = range(len(ur_mean)) if u_filter is None else u_filter
    if relative:
        sim_low = (ur_range[0][u_filter]
                   - ur_mean[u_filter])*100/ur_mean[u_filter]
        sim_high = (ur_range[1][u_filter]
                    - ur_mean[u_filter])*100/ur_mean[u_filter]
    else:
        sim_low = ur_range[0]
        sim_high = ur_range[1]
    ax.fill_between(
        rbin_centres[u_filter],sim_low,sim_high,
        color=seabornColormap[0],label="$Simulation, 68\\% interval$",alpha=0.5
    )
    u = 1 - np.cbrt(1 + Delta_mean)
    for k in range(len(alphas_list)):
        model_val = semi_analytic_model(
            u[u_filter],alphas_list[k],z=z,Om=Om
        )
        if relative:
            model_plot = (model_val - ur_mean[u_filter])*100/ur_mean[u_filter]
        else:
            model_plot = model_val
        ax.plot(
            rbin_centres[u_filter],model_plot,
            color=seabornColormap[k],linestyle=':',
            label="Empirical Model, $N = " + ("%.2g" % (k+2)) + "$"
        )
    plt.xlabel('$r/r_{\\mathrm{eff}}$')
    if relative:
        plt.ylabel('Percentage Error')
    else:
        plt.ylabel('$v_r/r [h\\mathrm{kms}^{-1}\\mathrm{Mpc}^{-1}]$')
    ax.set_ylim(ylim)
    plt.legend(**legend_params)
    if savename is not None:
        plt.savefig(savename)
    plt.show()


def plot_alphas(
        alpha_vec,
        ylim=[-10,10],
        savename=None
    ):
    # Plot of alphas:
    plt.clf()
    for n, alpha in enumerate(alpha_vec):
        plt.plot(
            range(2 + n,len(alpha_vec)+2),alpha,color=seabornColormap[n],
            marker='x',linestyle="--",label = f"$\\alpha_{n+2}$"
        )
    plt.xlabel('$N$')
    plt.ylabel('$\\alpha_n$')
    #plt.yscale('log')
    plt.ylim(ylim)
    plt.axhline(0.0,linestyle=":",color='k')
    plt.legend(frameon=False)
    if savename is not None:
        plt.savefig(savename)
    plt.show()



def plot_rho_real(
        rvals,
        delta_func,
        Delta_func,
        savename=None
    ):
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(rvals,delta_func(rvals),label='$\\delta(r)$')
    ax.plot(rvals,Delta_func(rvals),label='$\\Delta(r)$')
    ax.set_xlabel('r [$\\mathrm{Mpc}h^{-1}$]')
    ax.set_ylabel('$\\rho(r)$')
    plt.legend(frameon=False)
    if savename is not None:
        plt.savefig(savename)
    plt.show()




