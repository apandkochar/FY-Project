import cv2
import numpy as np
import os

image_directory = '/Users/admin/Downloads/91000201_4_1c2562b7c5c04d4c93845072ab4f75f4.png'

image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

outliers = 0
minn = 9999999
maxx = 0
for image_file in image_files:
    # Construct the full file path
    image_path = os.path.join(image_directory, image_file)
    
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is read correctly
    if img is None:
        print(f"Could not open or find the image: {image_file}")
        continue
    
    # Calculate the Laplacian variance
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    if laplacian_var > 300:
        outliers += 1

    if laplacian_var < minn:
        minn = laplacian_var

    if laplacian_var > maxx:
        maxx = laplacian_var
    
    cv2.waitKey(0)

print("Number of blur images that have laplacian variance more than threshold value (300) : ",outliers)
print("Lowest laplacian variance value  : ",minn)
print("Highest laplacian variance value  : ",maxx)
cv2.destroyAllWindows()