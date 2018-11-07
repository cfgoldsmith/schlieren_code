"""
Shock Tube Reactor

Implementation of shock tube governing equations derived as derived by Franklin
Goldsmith.

--------------------------------------------------------------------------------

Copyright (c) 2016 Raymond L. Speth

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import division

import cantera as ct
import numpy as np

def calc_postshock(gas, U1, rtol=1e-10):
    """
    Calculates frozen post-shock state for a specified shock velocity. The state
    of the gas object will reflect the post-shock state. Returns the post-shock
    velocity.

    :param gas:
        Solution object in the initial, pre-shock state
    :param U1:
        shock speed [m/s]
    """

    r1 = gas.density
    V1 = 1/r1
    P1 = gas.P
    T1 = gas.T
    H1 = gas.enthalpy_mass

    i = 0
    # Initial guess
    V = V1/5
    P = P1 + r1*(U1**2)*(1-V/V1)
    T = T1*P*V/(P1*V1)
    gas.TD = T, 1/V
    FH = FP = 1e8

    while(abs(FH/(gas.cp * T)) > rtol or abs(FP/gas.P) > rtol):
        i += 1
        if i == 500:
            raise Exception("shk_calc did not converge for U = %s" % U1)

        # Calculate FH, FP, and their derivatives
        r2 = gas.density
        w2s = U1**2 * (r1/r2)**2
        FH = gas.enthalpy_mass + 0.5*w2s - (H1 + 0.5*U1**2)
        FP = gas.P + r2*w2s - (P1 + r1*U1**2)

        DFHDT = gas.cp_mass
        DFPDT = gas.P / gas.T
        DFHDV = U1**2 * r1**2 / gas.density
        DFPDV = - gas.P * gas.density + U1**2 * r1**2

        # Construct Jacobian and solve
        J = DFHDT*DFPDV - DFPDT*DFHDV
        b = [DFPDV, -DFHDV, -DFPDT, DFHDT]
        a = [-FH, -FP]
        deltaT = (b[0]*a[0]+b[1]*a[1])/J
        deltaV = (b[2]*a[0]+b[3]*a[1])/J

        # limit changes in temperature and volume
        DTM = 0.2*T
        if abs(deltaT) > DTM:
            deltaT = DTM*deltaT/abs(deltaT)

        if V + deltaV > V1:
            DVM = 0.5*(V1 - V)
        else:
            DVM = 0.2*V
        if abs(deltaV) > DVM:
            deltaV = DVM*deltaV/abs(deltaV)

        # Apply increments to T and V
        T += deltaT
        V += deltaV
        gas.TD = T, 1/V

    return U1*r1 / gas.density


class ReactorOde(object):
    def __init__(self, gas, L, As, A1, area_change):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.N = gas.n_species + 6
        self.L = L
        self.As = As
        self.A1 = A1
        self.n = 0.5
        self.Wk = self.gas.molecular_weights
        self.rho1 = gas.density
        self.delta_dA = 1 if area_change else 0

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [z, A, rho, v, T, tlab, Y_1, Y_2, ... Y_K]
        #                  0  1   2   3  4   5    6 ...
        z, A, rho, v, T, tlab = y[:6]
        self.gas.set_unnormalized_mass_fractions(y[6:])
        self.gas.TD = T, rho
        cp = self.gas.cp_mass
        Wmix = self.gas.mean_molecular_weight
        hk = self.gas.partial_molar_enthalpies
        wdot = self.gas.net_production_rates

        beta = v**2 * (1.0/(cp*T) - Wmix / (ct.gas_constant * T))
        xi = max(z / self.L, 1e-10)

        ydot = np.zeros(self.N)
        ydot[0] = v # dz/dt
        ydot[1] = (self.delta_dA * v * self.As*self.n/self.L
                   * xi**(self.n-1.0)/(1.0-xi**self.n)**2.0) # dA/dt
        ydot[6:] = wdot * self.Wk / rho # dYk/dt
        ydot[2] = 1/(1+beta) * (sum((hk/(cp*T) - Wmix) * wdot)
                                - rho*beta/A * ydot[1]) # drho/dt
        ydot[3] = - v * (ydot[2]/rho + ydot[1]/A) # dv/dt
        ydot[4] = - (np.dot(wdot, hk)/rho + v*ydot[3]) / cp # dT/dt
        ydot[5] = self.rho1*self.A1 / (rho*A) # dt_lab/dt

        return ydot

    def drhodz(self, t, y):
        z, A, rho, v, T, tlab = y[:6]
        self.gas.set_unnormalized_mass_fractions(y[6:])
        self.gas.TD = T, rho
        cp = self.gas.cp_mass
        Wmix = self.gas.mean_molecular_weight
        hk = self.gas.partial_molar_enthalpies
        wdot = self.gas.net_production_rates

        beta = v**2 * (1.0/(cp*T) - Wmix / (ct.gas_constant * T))
        xi = max(z / self.L, 1e-10)

        dAdt = (self.delta_dA * v * self.As*self.n/self.L
               * xi**(self.n-1.0)/(1.0-xi**self.n)**2.0) # dA/dt
        return 1/v/(1+beta) * (sum((hk/(cp*T) - Wmix) * wdot) - rho*beta/A * dAdt)
