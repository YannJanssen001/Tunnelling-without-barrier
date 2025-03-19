# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 19:53:38 2025

@author: k21071708
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import sympy as sp
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import timeit

start = timeit.default_timer()

#Physical constants:
EAU = 27.2114
IAU = 3.5e16
LAU = 0.052918
TAU = 2.419e-17
alpha = 1. /137
c = 1/alpha

#Field configurations:
wavelength = 800 #e-9 #Wavelength in nm
Int_0 = 4e14 #Intensity in W/cm2
Ip = 0.5 #* 13.5984 # Hydrogen gas target ionization potential in eV

omega = 2 * np.pi * LAU * c / wavelength  # Angular frequency
TC = 2 * np.pi / omega  # Optical cycle period
E_0 = np.sqrt(Int_0 / IAU)  #field strength

#Defining electric field E(t):
t_list = np.linspace(-TC, TC, 200)  #time list

#two-colour beam, with mixing angle θ and phase shift φ
def beam_TC(theta, phi, r, s):
    Int_1 = Int_0 * (np.cos(theta))**2  #Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta))**2  #Intensity of beam 1
    omega_1 = r * omega  #Frequency of beam 1
    omega_2 = s * omega  #Frequency of beam 1

    E_01 = np.sqrt(Int_1 / IAU)  #field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  #field strength of beam 2

    e_field = []  #create empty lists to store electric field values for each point in time

    for i in t_list:
        beam_1 = E_01 * np.sin(omega_1 * i)
        beam_2 = E_02 * np.sin((omega_2 * i) + phi)
        total_beam = beam_1 + beam_2

        e_field.append(total_beam)

    return e_field

#Defining vector potential A(t):
def vector_potential_TC(theta, phi, r, s):
    Int_1 = Int_0 * (np.cos(theta))**2  #Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta))**2  #Intensity of beam 1
    omega_1 = r * omega  #Frequency of beam 1
    omega_2 = s * omega  #Frequency of beam 1

    E_01 = np.sqrt(Int_1 / IAU)  #field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  #field strength of beam 2

    vector_pot = []  
    
    for i in t_list:
        A_1 = (E_01 / omega_1) * np.cos(omega_1 * i)
        A_2 = (E_02 / omega_2) * np.cos((omega_2 * i) + phi)
        A_total = A_1 + A_2

        vector_pot.append(A_total)

    return vector_pot

#Makes a function of electric field amplitude in terms of only θ:
def field_strength(theta):
    Int_1 = Int_0 * (np.cos(theta))**2  #Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta))**2  #Intensity of beam 1

    E_01 = np.sqrt(Int_1 / IAU)  #field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  #field strength of beam 2

    return [E_01, E_02]

#initialize parameters
omega_1 = omega  #Frequency of beam 1
omega_2 = 2 * omega  #Frequency of beam 2

# Define the range of parameters
theta_values = np.linspace(np.radians(40), np.radians(60), 20)  # theta from 0 to pi
phi_values = np.linspace(0, np.pi, 20)  # phi from 0 to pi
p_values = np.linspace(-2.2, 2.2, 20)  # p from -2.0 to 2.0

# Initialize lists to store the configurations where E(ts) ≈ 0
zero_field_configs = []

# Iterate over all combinations of theta, phi, and p
for theta in theta_values:
    for phi in phi_values:
        for p in p_values:
            # Calculate the electric field at the saddle points
            E_01 = field_strength(theta)[0]
            E_02 = field_strength(theta)[1]
            
            # Calculate the saddle points for the current p
            ts = []
            
            def action_drv_TC(t_arr):
                t = t_arr[0] + t_arr[1]*1j
        
                dS_dt_TC_real = np.real(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #dS/dt = Ip + (p + A(t))^2 
                dS_dt_TC_imag = np.imag(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #where A(t) = E_0/w * cos(wt + phi)
    
                return np.array([dS_dt_TC_real, dS_dt_TC_imag])
        
            for m in np.linspace(0, TC, 10):
                for n in np.linspace(0, TC/2, 10):
                    saddles = fsolve(action_drv_TC, np.array([m, n]), xtol=1e-8)
                    saddles = np.round(saddles, 3)
                    if saddles[0] > 100-TC and saddles[0] < 100:
                        if saddles[1] > 0 and saddles[1] < TC/2:
                            if np.linalg.norm(action_drv_TC(saddles)) < 0.1:
                                ts.append(saddles)
                
            saddle_points = [complex(t[0], t[1]) for t in ts]
            saddle_points = np.unique(np.round(saddle_points, 3))
            
            #print(saddle_points)
            ts_B = np.real(saddle_points[1])  #keep only saddle point B
            
            # Calculate the electric field at saddle point B
            e_field_at_B = E_01 * np.sin(omega_1 * ts_B) + E_02 * np.sin((omega_2 * ts_B) + phi)

            if np.abs(e_field_at_B) < 1e-3:
                zero_field_configs.append((theta, phi, p))
                
# Convert the list of configurations to a numpy array
zero_field_configs = np.array(zero_field_configs)

# Check if there are any valid configurations
if zero_field_configs.size > 0:
    # Plot the 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(zero_field_configs[:, 0], zero_field_configs[:, 1], zero_field_configs[:, 2], c='r', marker='o')
    
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Momentum (p)')
    
    plt.show()
else:
    print("No configurations found where E(ts) ≈ 0.")


stop = timeit.default_timer()
print('Time: ', stop - start)

