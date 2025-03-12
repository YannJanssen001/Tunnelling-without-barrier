# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import sympy as sp
from scipy.optimize import fsolve

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
theta = np.pi/3
phi = np.pi/4 

E_01 = field_strength(theta)[0]
E_02 = field_strength(theta)[1]

omega_1 = omega  #Frequency of beam 1
omega_2 = 2 * omega  #Frequency of beam 2    

#Create list of momentum values 
momentum = np.linspace(-2.0, 2.0, 100)
    
#Collect the saddle points for A,B,C,D separately in order to plot 4 lines
saddles_A = []    
saddles_B = []
saddles_C = []
saddles_D = []

#Calculate the 4 saddle points for each p:
for p in momentum:
    
    ts = []
    
    def action_drv_TC(t_arr):
        t = t_arr[0] + t_arr[1]*1j

        dS_dt_TC_real = np.real(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #dS/dt = Ip + (p + A(t))^2 
        dS_dt_TC_imag = np.imag(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #where A(t) = E_0/w * cos(wt + phi)

        return np.array([dS_dt_TC_real, dS_dt_TC_imag])

    for m in np.linspace(0, TC, 10):                             #calculate saddle point for every initial guess in timespace
        for n in np.linspace(0, TC/2, 10):
            #print(m+1j*n)
            saddles = fsolve(action_drv_TC, np.array([m,n]), xtol=1e-8)
            saddles = np.round(saddles, 3)

            if saddles[0] > 0 and saddles[0] < TC:                                   #only keep saddle points with positive real/imag times
                if saddles[1] > 0 and saddles[1] < TC/2:
                    if np.linalg.norm(action_drv_TC(saddles)) < 0.1: # 1 is a numerical tolerance set heuristically at the monochromatic fields
                        ts.append(saddles)

    saddle_points = [complex(t[0], t[1]) for t in ts]            # Convert (real, imag) pairs to complex numbers
    saddle_points = np.unique(np.round(saddle_points, 3))        # Round to 6 d.p., then remove duplicates
    #print(saddle_points)
    
    saddles_A.append(np.real(saddle_points[0]))
    saddles_B.append(np.real(saddle_points[1]))
    saddles_C.append(np.real(saddle_points[2]))
    saddles_D.append(np.real(saddle_points[3]))
    
    e_field_A = []
    e_field_B = []
    e_field_C = []
    e_field_D = []

    for a in saddles_A:
        beam_1 = E_01 * np.sin(omega_1 * a)
        beam_2 = E_02 * np.sin((omega_2 * a) + phi)
        total_beam = beam_1 + beam_2

        e_field_A.append(total_beam)
        
    for b in saddles_B:
        beam_1 = E_01 * np.sin(omega_1 * b)
        beam_2 = E_02 * np.sin((omega_2 * b) + phi)
        total_beam = beam_1 + beam_2

        e_field_B.append(total_beam)
        
    for c in saddles_C:
        beam_1 = E_01 * np.sin(omega_1 * c)
        beam_2 = E_02 * np.sin((omega_2 * c) + phi)
        total_beam = beam_1 + beam_2

        e_field_C.append(total_beam)
        
    for d in saddles_D:
        beam_1 = E_01 * np.sin(omega_1 * d)
        beam_2 = E_02 * np.sin((omega_2 * d) + phi)
        total_beam = beam_1 + beam_2

        e_field_D.append(total_beam)

#print(np.real(saddles_A))
#print(e_field_A)


#Plot evolution of saddle points for varying momentum p
fig, ax = plt.subplots()
ax.set_xlabel("p")
ax.set_ylabel("Re(ts(p))")

ax.plot(momentum, saddles_A, color="red", label="A")
ax.plot(momentum, saddles_B, color="green", label="B")
ax.plot(momentum, saddles_C, color="blue", label="C")
ax.plot(momentum, saddles_D, color="orange", label="D")

# Draw a horizontal line at y = 0 to represent the x-axis
ax.axhline(0, color='grey', linewidth=0.8)



#Plot evolution of electric field at each saddle point for varying momentum p
fig, ax = plt.subplots()
ax.set_xlabel("p")
ax.set_ylabel("E(Re(ts(p)))")

# Draw a horizontal line at y = 0 to represent the x-axis
ax.axhline(0, color='grey', linewidth=0.8)

ax.plot(momentum, e_field_A, color="red", label="A")
ax.plot(momentum, e_field_B, color="green", label="B")
ax.plot(momentum, e_field_C, color="blue", label="C")
ax.plot(momentum, e_field_D, color="orange", label="D")






'''
#TRAJECTORIES:
    
# Create a figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlabel("Re(ωt)")
ax.set_ylabel("Re(x(Re(ωt)))")

# Plot the electric field
times = np.linspace(0.0, 2 * TC, 200)  # 200 time points
electric_field = [i*500 for i in beam_TC(theta, phi, 1, 2)]  #theta, phi, r, s

ax.plot(times, electric_field, color="black", label="Electric Field")

# Define labels and colors for each trajectory
labels = ["A", "B", "C", "D"]
colors = ["red", "green", "blue", "orange"]

#Plot the trajectories
for idx, sad in enumerate(saddle_points):
    real_times = np.linspace(np.real(sad), 2*TC, 100)

    positions = []
    
    for t in real_times:
        integrated_A_t = (E_01/omega_1**2) * np.sin(omega_1*t) + (E_02/omega_2**2) * np.sin(omega_2*t + phi)
        integrated_A_ts = (E_01/omega_1**2) * np.sin(omega_1*sad) + (E_02/omega_2**2) * np.sin(omega_2*sad + phi)
        integrated_A = integrated_A_t - integrated_A_ts

        final_x = p*(t - sad) + integrated_A
        positions.append(final_x)

    ax.plot(real_times, np.real(positions), linestyle="solid", label=f"Trajectory {labels[idx]}", color=colors[idx])

# Draw a horizontal line at y = 0 to represent the x-axis
ax.axhline(0, color='grey', linewidth=0.8)


# Set x-axis to visually represent π * t with labels as π, 2π, 3π, etc.
ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f'{t*np.pi}π'))  # Format labels as π * t


plt.title('θ = 45, φ = 90')
plt.legend()
plt.show()
'''