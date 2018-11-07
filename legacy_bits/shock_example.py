"""
Shock Tube Reactor example
"""

from __future__ import division

import cantera as ct
import numpy as np
import scipy.integrate
import shockreactor

mech = 'furan.cti'
# define the initial state of the test gas:
T1 = 299.75 #Kelvin
P1 = 0.006907895 * ct.one_atm  # Pa
X1 = 'kr:0.98, mf2:0.02' #initial composition of the driven gas, in mole fractions

# define the shock velocity
UI = 9.38e2 #Incident Shock Speed (m/s)

gas = ct.Solution('furan.cti')
gas.TPX = T1, P1, X1
rhoI = gas.density

# compute the frozen state behind the incident shock
U1 = shockreactor.calc_postshock(gas, UI)

A1 = 0.2
As = 0.2
L = 0.1

# Initial condition
y0 = np.hstack((0.0, A1, gas.density, U1, gas.T, 0.0, gas.Y))

# Set up objects representing the ODE and the solver
ode = shockreactor.ReactorOde(gas, L, As, A1, False)
solver = scipy.integrate.ode(ode)
solver.set_integrator('vode', method='bdf', with_jacobian=True, nsteps=1000)
#solver.set_integrator('lsoda', with_jacobian=True, nsteps=10000)
solver.set_initial_value(y0, 1e-20)

# Integrate the equations, keeping T(t) and Y(k,t)
t_end = 50e-6
t_out = [0.0]
T_out = [gas.T]
Y_out = [gas.Y]
X_out = [gas.X]
v_out = [U1]
A_out = [A1]
drhodz = [ode.drhodz(0.0, y0)]
rho_out = [gas.density]
z_out = [0.0]

dt = t_end/1000
while solver.successful() and solver.t < t_end:
    try:
        solver.integrate(solver.t + dt)
    except Exception as e:
        print(e)
        break
    t_out.append(solver.t)
    z_out.append(solver.y[0])
    T_out.append(solver.y[4])
    Y_out.append(solver.y[6:])
    A_out.append(solver.y[1])
    v_out.append(solver.y[3])
    rho_out.append(solver.y[2])
    gas.Y = solver.y[6:]
    X_out.append(gas.X)
    drhodz.append(ode.drhodz(solver.t, solver.y))

z_out = np.array(z_out)
Y_out = np.array(Y_out).T
X_out = np.array(X_out).T

# Plot the results
try:
    import matplotlib.pyplot as plt

    k = 'mf2' # species to plot
    plt.subplot(2,1,1)
    L1 = plt.plot(t_out, T_out, color='r', label='T', lw=2)
    plt.xlabel('time (s)')
    plt.ylabel('Temperature (K)')
    plt.twinx()
    L2 = plt.plot(t_out, X_out[gas.species_index(k)], label=k, lw=2)
    plt.ylabel('Mole Fraction')
    plt.legend(L1+L2, [line.get_label() for line in L1+L2], loc='upper right')

    plt.subplot(2,1,2)
    plt.plot(z_out, drhodz, lw=2)
    plt.plot(0.5*(z_out[1:]+z_out[:-1]), np.diff(rho_out)/np.diff(z_out), 'y--')
    plt.xlabel('z [m]')
    plt.ylabel(r'$\partial \rho / \partial z$')
    plt.show()

except ImportError:
    print('Matplotlib not found. Unable to plot results.')
