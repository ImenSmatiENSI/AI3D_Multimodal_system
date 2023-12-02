import math
import numpy as np
import cmath
import pdb
import pandas as pd
import numpy as np
import pandas as pd
from functools import reduce
import math
from scipy import misc
import matplotlib.pyplot as plt


# Function to generate a range of numbers between a and b with step size s
def any_number_range(a,b,s=1):
    result = []
    if (a == b):
        result.append(int(a))
        #print(type(result))
    else:
        mx = max(a,b)
        mn = min(a,b)
       
        # inclusive upper limit. If not needed, delete '+1' in the line below
        while(mn <= mx):
            # if step is positive we go from min to max
            if s > 0:
                result.append(int(mn))
                mn += s
            # if step is negative we go from max to min
            if s < 0:
                result.append(int(mx))
                mx += s
    #print(result)
    return result


# Function to rotate a 3D contour in the specified ranges
def rotate3D_contour_bs(cont, phi_range_min, phi_range_max, ksi1_range_min, ksi1_range_max, ksi2_range_min, ksi2_range_max, batch_size):
    x = cont[0:120]
    y = cont[120:240]
    
    jj = complex(0,1)
    
    # Initialization of arrays for storing lower and upper bounds of x and y coordinates
    x_lw = np.zeros(120,)
    x_up = np.zeros(120,)
    lower_final_x = np.zeros(120,)
    upper_final_x = np.zeros(120,)
    
    y_lw = np.zeros(120,)
    y_up = np.zeros(120,)
    lower_final_y = np.zeros(120,)
    upper_final_y = np.zeros(120,)
    
    Upper_bound_cont = np.zeros(240,)
    Lower_bound_cont = np.zeros(240,)
    
    # Calculation of step sizes for rotation angles
    step_phi = (phi_range_max - phi_range_min) / batch_size
    step_ksi1 = (ksi1_range_max - ksi1_range_min) / batch_size
    step_ksi2 = (ksi2_range_max - ksi2_range_min) / batch_size
    
    # Loop over each batch
    for h in any_number_range(0, batch_size):
        # Calculation of rotation angles for the current batch
        degre_phi = phi_range_min + h * step_phi 
        degre1_phi = phi_range_min + (h + 1) * step_phi
        
        degre_ksi1 = ksi1_range_min + h * step_ksi1 
        degre1_ksi1 = ksi1_range_min + (h + 1) * step_ksi1 
        
        degre_ksi2 = ksi2_range_min + h * step_ksi2
        degre1_ksi2 = ksi2_range_min + (h + 1) * step_ksi2

        # Convert rotation angles from degrees to radians  
        phis_lw = np.deg2rad(degre_phi)
        phis_up = np.deg2rad(degre1_phi)

        ksi1s_lw = np.deg2rad(degre_ksi1)
        ksi1s_up = np.deg2rad(degre1_ksi1)

        ksi2s_lw = np.deg2rad(degre_ksi2)
        ksi2s_up = np.deg2rad(degre1_ksi2)
                
        # Calculation of complex coefficients for rotation
        a_val_lw = np.zeros(batch_size, dtype=complex)
        b_val_lw = np.copy(a_val_lw)
                
        a_val_up = np.zeros(batch_size, dtype=complex)
        b_val_up = np.copy(a_val_lw)
                
        a_val_lw = np.cos(phis_lw / 2) * np.exp(jj * (ksi1s_lw + ksi2s_lw) / 2)
        b_val_lw = jj * np.sin(phis_lw / 2) * np.exp(jj * (ksi1s_lw - ksi2s_lw) / 2)
        
        a_val_up = np.cos(phis_up / 2) * np.exp(jj * (ksi1s_up + ksi2s_up) / 2)
        b_val_up = jj * np.sin(phis_up / 2) * np.exp(jj * (ksi1s_up - ksi2s_up) / 2)
        
        a_min = min(a_val_lw, a_val_up)
        b_min = min(b_val_lw, b_val_up)
        a_max = max(a_val_lw, a_val_up)
        b_max = max(b_val_lw, b_val_up)
        
        # Initialization of arrays for storing rotated coordinates
        z = np.zeros((120,), dtype=complex)
        z_lw = np.zeros((120,), dtype=complex)
        z_up = np.zeros((120,), dtype=complex)
        z_min = np.zeros((120,), dtype=complex)
        z_max = np.zeros((120,), dtype=complex)
     
        # Loop over each point in the contour
        for i in range(0, len(x)):
            z[i] = x[i] + jj * y[i]
            z_lw[i] = (a_min * z[i] + b_min) / (-b_min.conjugate() * z[i] + a_min.conjugate())   
            z_up[i] = (a_max * z[i] + b_max) / (-b_max.conjugate() * z[i] + a_min.conjugate())   
            
            z_min[i] = min(z_lw[i], z_lw[i])
            z_max[i] = max(z_up[i], z_up[i])
            
            y_lw[i] = z_min[i].imag
            x_lw[i] = z_min[i].real
            y_up[i] = z_max[i].imag
            x_up[i] = z_max[i].real
            
            # Initial values of bounds are set when h == 0        
            if  h == 0:
                lower_final_x = x_lw
                lower_final_y = y_lw
                upper_final_x = x_up
                upper_final_y = y_up
                
            else:
                # Update bounds for subsequent batches
                for j in range(len(x)):
                    lower_final_x[j] = min(lower_final_x[j], x_lw[j], x_up[j])
                    lower_final_y[j] = min(lower_final_y[j], y_lw[j], y_up[j])  
                    upper_final_x[j] = max(upper_final_x[j], x_lw[j], x_up[j])
                    upper_final_y[j] = max(upper_final_y[j], y_lw[j], y_up[j]) 


    # Concatenate lower and upper bounds for both x and y coordinates
    Lower_bound_final = np.concatenate([lower_final_x, lower_final_y]) 
    Upper_bound_final = np.concatenate([upper_final_x, upper_final_y]) 
    
    # Display the plot
    '''
    plt.plot(x,y,label="Contour_final")        
    plt.plot(lower_final_x,lower_final_y,label="Lower Contour")
    plt.plot(upper_final_x,upper_final_y,label="Upper Contour")
    plt.legend(bbox_to_anchor=(1.05,1),loc='upper left',borderaxespad=0.)
    #plt.legend()
    plt.show()
    '''


    return Lower_bound_final

