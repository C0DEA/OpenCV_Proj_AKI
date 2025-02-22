import cv2
import numpy as np

def gpu_image_processing():
    # Read an image
    img = cv2.imread('image.jpg')

    # Create GPU matrices
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    # Create GPU-accelerated image processing objects
    gpu_blur = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.0
    )
    gpu_gray = cv2.cuda.createGpuMat((img.shape[0], img.shape[1]), cv2.CV_8UC1)

    # Process on GPU
    # Apply Gaussian blur
    blurred_gpu = gpu_blur.apply(gpu_img)

    # Convert to grayscale
    cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY, gpu_gray)

    # Download results back to CPU
    result_blur = blurred_gpu.download()
    result_gray = gpu_gray.download()

    return result_blur, result_gray

# Alternative approach using CUDA-enabled functions
def cuda_operations():
    img = cv2.imread('image.jpg')

    # Convert to grayscale using CUDA
    gray = cv2.cuda.cvtColor(
        cv2.cuda_GpuMat(img),
        cv2.COLOR_BGR2GRAY
    ).download()

    # Edge detection using CUDA
    edges = cv2.cuda.createCannyEdgeDetector(
        100, 200
    ).detect(
        cv2.cuda_GpuMat(gray)
    ).download()

    return gray, edges
