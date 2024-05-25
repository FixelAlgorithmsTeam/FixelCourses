import numpy as np
import cupy as cp
import cupyx.scipy.signal

import time
from scipy.signal import convolve2d

# Create a random image of size 1024x1024
np_image = np.random.rand(1024, 1024).astype(np.float32)

# Create a simple 3x3 edge detection kernel
kernel = np.array([[1, 0, -1],
                   [0, 0, 0],
                   [-1, 0, 1]], dtype=np.float32)

# Convert numpy arrays to cupy arrays
cp_image = cp.asarray(np_image)
cp_kernel = cp.asarray(kernel)

# Function to perform convolution using NumPy
def numpy_convolution(image, kernel):
    return convolve2d(image, kernel, mode='same', boundary='wrap')

# Function to perform convolution using CuPy
def cupy_convolution(image, kernel):
    return cp.asnumpy(cupyx.scipy.signal.convolve2d(image, kernel, mode='same', boundary='wrap'))

# Timing the CPU (NumPy)
start_time = time.time()
for _ in range(100):
    np_result = numpy_convolution(np_image, kernel)
cpu_time = time.time() - start_time
print("CPU time: {:.5f} seconds".format(cpu_time))

# Timing the GPU (CuPy)
cp.cuda.Stream.null.synchronize()  # Wait for GPU sync
start_time = time.time()
for _ in range(100):
    cp_result = cupy_convolution(cp_image, cp_kernel)
cp.cuda.Stream.null.synchronize()  # Wait for GPU sync again after completion
gpu_time = time.time() - start_time
print("GPU time: {:.5f} seconds".format(gpu_time))

# Output the speedup
print("GPU is {:.2f} times faster than CPU.".format(cpu_time / gpu_time))
