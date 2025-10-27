#!/bin/env python3
'''
Recreation of a project I did in Matlab back in 1998 for a Numerical Analysis class in my B.S
degree

I do not have the original Matlab code. So I pulled python code from google and several
sites discussing SVD compression.

In the original class we were given the Mona Lisa in a grayscale format. In this code I use the
JPG available from wikipedia and the Full size or 512p image. A quick bit of code from google
search allows the conversion from full color to grayscale in numpy.

In our assigned project we had to print the original, Print 3 reduced version and plot the Sigma graph. Then turn in our printed images and the Matlab code.

'''
import sys
import datetime
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def rgb_to_grayscale(rgb_image_array):
    """
    Converts an RGB image (NumPy array) to a grayscale image.

    Args:
        rgb_image_array (np.ndarray): A 3D NumPy array representing an RGB image
                                     with shape (height, width, 3).

    Returns:
        np.ndarray: A 2D NumPy array representing the grayscale image.
    """
    # Define the standard weights for converting RGB to grayscale
    # These weights account for human perception of color brightness
    weights = np.array([0.299, 0.587, 0.114])

    # Perform a dot product along the last axis (color channels)
    # This effectively calculates the weighted sum for each pixel
    grayscale_image_array = np.dot(rgb_image_array[..., :3], weights)

    return grayscale_image_array

if len(sys.argv) !=2:
    print("Usage: python3 image.py <imagefile>")
    sys.exit(1)
img = Image.open(sys.argv[1])
print("Loading image as numpy array...")
img_array = np.array(img)

print("Converting to grayscale...")
grey_array= rgb_to_grayscale(img_array)

# Create a 2D array representing a grayscale image
plt.imshow(grey_array, cmap='gray')
plt.title("Grayscale Image")
#plt.show()
print(f"Image Size: {grey_array.nbytes}")
'''
This is where the original assignment would have begun. 
'''

'''
Since I long ago forgot how to do this.  I looked up the SVD method from the website below:

https://www.statology.org/how-to-perform-matrix-factorization-using-svd-with-numpy/

Understanding Singular Value Decomposition
SVD is a method that breaks a matrix into three distinct parts. The formula for SVD is:

𝐴=𝑈Σ𝑉t
'''

A=grey_array
print("Calculating 𝑈Σ𝑉t matrixes")
U, S, Vt = np.linalg.svd(A)
print(f"Matrix Total Byte Size: {U.nbytes+S.nbytes+Vt.nbytes}")

print("\nU Matrix (Left Singular Vectors):\n", U)
print("\nSingular Values:\n", S)
print("\nVt Matrix (Right Singular Vectors Transposed):\n", Vt)
print(f"\n\n{50*'='}")
print(f"U.shape:{U.shape}, S.shape:{S.shape}, V.shape: {Vt.shape}")


Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, S)
print("\nSigma (Diagonal Matrix of Singular Values):\n", Sigma)

A_reconstructed = np.dot(U, np.dot(Sigma, Vt))
print("Reconstructed Matrix:\n", A_reconstructed)
plt.imshow(A_reconstructed, cmap='gray')
plt.title("Reconstructed Image")
plt.savefig("Reconstructed_mona.png")
plt.clf()
print("\nReconstructed image saved")
print(f"{50*'='}")


'''
https://medium.com/@sudhanshumukherjeexx/how-to-use-singular-value-decomposition-for-image-compression-7d45882f9f23
'''
plt.clf()
plt.plot(range(1, len(S) + 1), S, 'r-')
plt.xlabel("Rankings")
plt.ylabel("Singular Values")
plt.title("Singular Values versus their Rankings")
plt.savefig("Singular_values_vs_rankings.png")

k_values = [10, 50, 100, 256]
k_values.append(len(S))
plt.clf()

for i in range(len(k_values)):
    Usub = U[:, :k_values[i]]
    Ssub = np.diag(S[:k_values[i]])
    Vtsub = Vt[:k_values[i], :]
    low_rank = Usub @ Ssub @ Vtsub
    tbytes=Usub.nbytes+Ssub.nbytes+Vtsub.nbytes
    print(f"K_value Matrix Total Byte Size:{k_values[i]} {tbytes}")
    print(f"Image Size: {low_rank.nbytes}")
    print(f"Time to trasmit using Voyager Baud Rate(H:M:S): {datetime.timedelta(seconds = tbytes/20)}")
    print(f"{50*'-'}")
    plt.subplot(2,3,i+1),
    plt.imshow(low_rank, cmap='gray'),
    plt.title(f"For K value = {k_values[i]}")
    plt.savefig("Reconstruction_with_k_values.png")
#plt.show()

