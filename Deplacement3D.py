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

def any_number_range(a, b, s=1):
    # Function to generate a range of numbers between a and b with step size s
    result = []
    if (a == b):
        result.append(a)
    else:
        mx = max(a, b)
        mn = min(a, b)
        while(mn <= mx):
            if s > 0:
                result.append(mn)
                mn += s
            if s < 0:
                result.append(mx)
                mx += s
    return result

def Deplacement3D_contour_bs(cont, b1_min, b1_max, b2_min, b2_max, b3_min, b3_max, batch_size):
    # Main function for 3D contour transformations
    x = cont[0:120]
    y = cont[120:240]
    lx = x.shape
    ly = y.shape
    jj = complex(0, 1)

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

    # Calculation of step sizes for translation and scaling
    step_transx = abs(b1_max - b1_min) / batch_size
    step_transy = abs(b3_max - b3_min) / batch_size
    step_scale = abs(b2_max - b2_min) / batch_size
    print('step-b2', step_scale)

    for k in any_number_range(0, batch_size):
        # Calculation of translation and scaling parameters for the current batch
        transx_min = b1_min + (k * step_transx)
        transx_max = b1_min + ((k + 1) * step_transx)
        transy_min = b3_min + (k * step_transy)
        transy_max = b3_min + ((k + 1) * step_transy)
        scale_min = b2_min + (k * step_scale)
        scale_max = b2_min + ((k + 1) * step_scale)
        alpha_min = pow((1 + scale_min), (1 / 2))
        alpha_max = pow((1 + scale_max), (1 / 2))
        gamma_min = (transy_min + jj * transx_min) * (1 / alpha_min)
        gamma_max = (transy_max + jj * transx_max) * (1 / alpha_max)

        z = np.zeros((120,), dtype=complex)
        z_max = np.zeros((120,), dtype=complex)
        z_min = np.zeros((120,), dtype=complex)

        for i in range(0, len(x)):
            # Transformation of coordinates for the current batch
            z[i] = x[i] + jj * y[i]
            z_min[i] = ((1 / alpha_min) * z[i] + gamma_min) / alpha_min
            z_max[i] = ((1 / alpha_max) * z[i] + gamma_max) / alpha_max
            x_up[i] = z_max[i].real
            y_up[i] = z_max[i].imag
            x_lw[i] = z_min[i].real
            y_lw[i] = z_min[i].imag

            if k == 0:
                # Update initial bounds for the first batch
                lower_final_x[i] = x_lw[i]
                lower_final_y[i] = y_lw[i]
                upper_final_x[i] = x_up[i]
                upper_final_y[i] = y_up[i]
            else:
                # Update bounds for subsequent batches
                lower_final_x[i] = min(lower_final_x[i], x_lw[i])
                lower_final_y[i] = min(lower_final_y[i], y_lw[i])
                upper_final_x[i] = max(upper_final_x[i], x_up[i])
                upper_final_y[i] = max(upper_final_y[i], y_up[i])
            
    # Concatenate lower and upper bounds for both x and y coordinates
    lower_bound_final = np.concatenate([lower_final_x, lower_final_y])
    Upper_bound_final = np.concatenate([upper_final_x, upper_final_y])

    # Display the plot
    #plt.plot(x,y,label="Contour")     
    #plt.plot(lower_final_x,lower_final_y,color= 'orange' ,label="Lower Contour")
    #plt.plot(upper_final_x,upper_final_y,color= 'green',label="Upper Contour")
    #plt.legend(bbox_to_anchor=(1.05,1),loc='upper left',borderaxespad=0.)
    #plt.legend()
    #plt.show()

    
    
    # Return the final upper and lower bounds
    return Upper_bound_final, lower_bound_final

