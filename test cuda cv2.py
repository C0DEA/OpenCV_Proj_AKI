import cv2
print("OpenCV version:", cv2.__version__)

print(cv2.getBuildInformation())

try:
    print("GPU devices recognized by OpenCV:", cv2.cuda.getCudaEnabledDeviceCount())
except AttributeError:
    print("No CUDA support in this OpenCV build.")

    