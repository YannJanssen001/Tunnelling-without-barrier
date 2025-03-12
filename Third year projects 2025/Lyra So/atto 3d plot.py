# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:12:37 2025

@author: k21071708
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import sympy as sp
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Physical constants:
EAU = 27.2114
IAU = 3.5e16
LAU = 0.052918
TAU = 2.419e-17
alpha = 1. / 137
c = 1 / alpha

# Field configurations:
wavelength = 800  # Wavelength in nm
Int_0 = 4e14  # Intensity in W/cm2
Ip = 0.5  # Hydrogen gas target ionization potential in eV

omega = 2 * np.pi * LAU * c / wavelength  # Angular frequency
TC = 2 * np.pi / omega  # Optical cycle period
E_0 = np.sqrt(Int_0 / IAU)  # Field strength

# Define electric field E(t):
t_list = np.linspace(-TC, TC, 200)  # Time list

# Two-colour beam, with mixing angle θ and phase shift φ
def beam_TC(theta, phi, r, s):
    Int_1 = Int_0 * (np.cos(theta)) ** 2  # Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta)) ** 2  # Intensity of beam 2
    omega_1 = r * omega  # Frequency of beam 1
    omega_2 = s * omega  # Frequency of beam 2

    E_01 = np.sqrt(Int_1 / IAU)  # Field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  # Field strength of beam 2

    e_field = []  # Create empty list to store electric field values

    for i in t_list:
        beam_1 = E_01 * np.sin(omega_1 * i)
        beam_2 = E_02 * np.sin((omega_2 * i) + phi)
        total_beam = beam_1 + beam_2
        e_field.append(total_beam)

    return e_field

# Define vector potential A(t):
def vector_potential_TC(theta, phi, r, s):
    Int_1 = Int_0 * (np.cos(theta)) ** 2  # Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta)) ** 2  # Intensity of beam 2
    omega_1 = r * omega  # Frequency of beam 1
    omega_2 = s * omega  # Frequency of beam 2

    E_01 = np.sqrt(Int_1 / IAU)  # Field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  # Field strength of beam 2

    vector_pot = []

    for i in t_list:
        A_1 = (E_01 / omega_1) * np.cos(omega_1 * i)
        A_2 = (E_02 / omega_2) * np.cos((omega_2 * i) + phi)
        A_total = A_1 + A_2
        vector_pot.append(A_total)

    return vector_pot

# Function to compute electric field amplitude in terms of θ:
def field_strength(theta):
    Int_1 = Int_0 * (np.cos(theta)) ** 2  # Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta)) ** 2  # Intensity of beam 2

    E_01 = np.sqrt(Int_1 / IAU)  # Field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  # Field strength of beam 2

    return [E_01, E_02]

# Initialize parameters
theta = np.pi / 4  # Mixing angle
phi_values = np.linspace(0, 2 * np.pi, 50)  # Range of phi values
momentum = np.linspace(-2.0, 2.0, 100)  # Momentum values

# Initialize arrays to store electric field values for each saddle point
e_field_A = np.zeros((len(momentum), len(phi_values)))
e_field_B = np.zeros((len(momentum), len(phi_values)))
e_field_C = np.zeros((len(momentum), len(phi_values)))
e_field_D = np.zeros((len(momentum), len(phi_values)))

# Loop over phi values
for idx, phi in enumerate(phi_values):
    E_01 = field_strength(theta)[0]
    E_02 = field_strength(theta)[1]

    omega_1 = omega  # Frequency of beam 1
    omega_2 = 2 * omega  # Frequency of beam 2

    # Calculate saddle points and electric fields for each momentum value
    for p_idx, p in enumerate(momentum):
        ts = []

        def action_drv_TC(t_arr):
            t = t_arr[0] + t_arr[1] * 1j
            dS_dt_TC_real = np.real(Ip + 0.5 * (p + (E_01 / omega_1) * np.cos(omega_1 * t) + (E_02 / omega_2) * np.cos(omega_2 * t + phi)) ** 2)
            dS_dt_TC_imag = np.imag(Ip + 0.5 * (p + (E_01 / omega_1) * np.cos(omega_1 * t) + (E_02 / omega_2) * np.cos(omega_2 * t + phi)) ** 2)
            return np.array([dS_dt_TC_real, dS_dt_TC_imag])

        for m in np.linspace(0, TC, 10):  # Calculate saddle point for every initial guess in timespace
            for n in np.linspace(0, TC / 2, 10):
                saddles = fsolve(action_drv_TC, np.array([m, n]), xtol=1e-8)
                saddles = np.round(saddles, 3)

                if saddles[0] > 0 and saddles[0] < TC:  # Only keep saddle points with positive real/imag times
                    if saddles[1] > 0 and saddles[1] < TC / 2:
                        if np.linalg.norm(action_drv_TC(saddles)) < 0.1:  # Numerical tolerance
                            ts.append(saddles)

        saddle_points = [complex(t[0], t[1]) for t in ts]  # Convert to complex numbers
        saddle_points = np.unique(np.round(saddle_points, 3))  # Remove duplicates

        # Compute electric field at saddle points
        if len(saddle_points) >= 4:
            e_field_A[p_idx, idx] = E_01 * np.sin(omega_1 * np.real(saddle_points[0])) + E_02 * np.sin(omega_2 * np.real(saddle_points[0]) + phi)
            e_field_B[p_idx, idx] = E_01 * np.sin(omega_1 * np.real(saddle_points[1])) + E_02 * np.sin(omega_2 * np.real(saddle_points[1]) + phi)
            e_field_C[p_idx, idx] = E_01 * np.sin(omega_1 * np.real(saddle_points[2])) + E_02 * np.sin(omega_2 * np.real(saddle_points[2]) + phi)
            e_field_D[p_idx, idx] = E_01 * np.sin(omega_1 * np.real(saddle_points[3])) + E_02 * np.sin(omega_2 * np.real(saddle_points[3]) + phi)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for p and phi
P, PHI = np.meshgrid(momentum, phi_values, indexing='ij')

# Plot surfaces for each saddle point
ax.plot_surface(P, PHI, e_field_A, color='red', label="A", alpha=0.6)
ax.plot_surface(P, PHI, e_field_B, color='green', label="B", alpha=0.6)
ax.plot_surface(P, PHI, e_field_C, color='blue', label="C", alpha=0.6)
ax.plot_surface(P, PHI, e_field_D, color='orange', label="D", alpha=0.6)

# Set labels
ax.set_xlabel("Momentum (p)")
ax.set_ylabel("Phase (phi)")
ax.set_zlabel("Electric Field (E)")

# Show plot
plt.show()