import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from skimage import measure

# Define the function to apply the process
def process_image(image_path, label):
    # Open and display the image
    image = Image.open(image_path)

    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    # Convert to grayscale using OpenCV
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Display the grayscale image
    #plt.imshow(gray_image, cmap='gray')
    #plt.axis('off')  # Hide axes for a cleaner view
    #plt.show()

    # Convert to binary image using thresholding
    _, binary_image = cv2.threshold(gray_image, 225, 255, cv2.THRESH_BINARY_INV)

    # Convert the binary image to the correct format (uint8)
    binary_image = (binary_image * 255).astype(np.uint8)

    # scikit-learn imaging contour finding, returns a list of found edges
    contours = measure.find_contours(binary_image, .8)

    # From which we choose the longest one
    if contours:
        contour = max(contours, key=len)
        print(f"Contour shape: {contour.shape}")

        # Display the contour on top of the binary image
        #plt.plot(contour[::,1], contour[::,0], linewidth=0.5)
        #plt.imshow(binary_image, cmap='Set3')
        #plt.show()

        # Resample and parameterize the contour (replace with your reparametrage_affine2 function)
        x_rep, y_rep, L = reparametrage_affine2(contour[::,1], -contour[::,0], 120)
        plt.plot(contour[::,1], -contour[::,0], linewidth=0.5)
        #plt.plot(x_rep,y_rep, label='contours reparamétré',  color='green')
        plt.scatter(x_rep,y_rep, label='contours reparamétré', color='red')
        plt.show()
        contours_final = np.concatenate((x_rep, y_rep))
        print(f"Final contour shape: {contours_final.shape}")
        return label, contours_final
    else:
        print(f"No contours found in image: {image_path}")
        return label, []

from numpy.core.multiarray import concatenate
import cv2
import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
#from scipy.integrate import cumtrapz
from scipy.integrate import cumulative_trapezoid

def reparametrage_affine2(X, Y, N):
    # Parameterize the curve using cubic splines
    n = len(X)
    t = np.linspace(0, 1, n)

    px = CubicSpline(t, X)
    py = CubicSpline(t, Y)

    # Resample the curve with specified number of points
    X1, Y1, L = abscisse_affine2(t, px, py, N)

    return X1, Y1, L

def abscisse_affine2(t, px, py, N):
    # Compute the derivatives of the splines
    dp_x = px.derivative()
    dp2_x = dp_x.derivative()

    dp_y = py.derivative()
    dp2_y = dp_y.derivative()

    # Compute the first and second derivatives
    X1 = dp_x(t)
    X2 = dp2_x(t)
    Y1 = dp_y(t)
    Y2 = dp2_y(t)

    # Compute the integrand for arc length
    F = np.abs(X1 * Y2 - Y1 * X2)**(1/3)
    I = cumulative_trapezoid(F, t, initial=0)

    # Compute the normalized arc length
    if np.max(I) == 0 or not np.isfinite(np.max(I)):
        s = np.zeros_like(t)
        L = 1
    else:
        s = I / np.max(I)
        L = s[-1]

    # Resample the curve based on normalized arc length
    unique_s, index = np.unique(s, return_index=True)
    out = np.interp(np.linspace(0, 1, N), unique_s, t[index])

    # Evaluate the splines at the resampled points
    X1_val = px(out)
    Y1_val = py(out)

    return X1_val/L, Y1_val/L, L


