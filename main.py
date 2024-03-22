import os
import pandas as pd
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def readdata(initpath) -> list:
    # Read annotation data from a .odgt file
    data = []
    filepath = os.path.join(initpath, "annotation_train.odgt")
    with open(filepath) as file:
        for line in file:
            data.append(json.loads(line))
    return data

def readimages(data, initpath) -> list:
    # Read and return images along with their annotation
    images = []
    for i in range(min(len(data), 5)):  # Limit to first 5 for demonstration
        ID = data[i]['ID']
        found = False
        for folder in ["CrowdHuman_train01", "CrowdHuman_train02", "CrowdHuman_train03"]:
            path = os.path.join(initpath, folder, "Images", f"{ID}.JPG")
            img = cv2.imread(path)
            if img is not None:
                images.append((img, data[i]))  # Pair image with its annotation
                found = True
                break
        if not found:
            print(f"Image {ID}.JPG not found in any directory.")
    return images

"""
Image Size and Objects Detected
"""
def compar_image_sizes_and_objects(images):
    # Compare image sizes with the number of objects detected
    for img, annotation in images:
        num_objects = len(annotation['gtboxes'])
        height, width = img.shape[:2]
        print(f"Image ID: {annotation['ID']}, Dimensions: {width}x{height}, Objects detected: {num_objects}")

"""
Gaussian Noise
"""
def add_gaussian_noise(image, mean=0, sigma=25):
    # Add Gaussian noise to an image
    height, width, _ = image.shape
    noise = np.random.normal(mean, sigma, (height, width, 3))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def augment_images_with_noise(images):
    # Apply Gaussian noise to a list of images
    return [add_gaussian_noise(image) for image in images]

"""
Display Images
"""
def displayimages(images) -> None:
    # Display each image in a list
    for i, img in enumerate(images):
        if img is not None:  # Check if image exists
            cv2.imshow(str(i), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Image at index {i} is None, skipping.")

"""
Histogram
"""
def pixelhistogram(images) -> None:
    # Display pixel intensity histogram for each image
    for img in images:
        vals = img.mean(axis=2).flatten()
        counts, bins = np.histogram(vals, range(257))
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        plt.xlim([-0.5, 255.5])
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Pixel Intensity Histogram')
        plt.show()

"""
Color Transformation
"""
def colortransformations(images) -> None:
    # Apply and display HSV transformation for each image
    for i, img in enumerate(images):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow(f"Image {i} - HSV", hsv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    initpath = "C:\\Users\\aleja\\PycharmProjects\\Human-Detection-in-Crowd-Images\\"  # Correct base path
    data = readdata(initpath)
    images_and_annotations = readimages(data, initpath)
    images = [img for img, _ in images_and_annotations]  # Extract images for processing
    compar_image_sizes_and_objects(images_and_annotations)
    displayimages(images)
    noisy_images = augment_images_with_noise(images)
    displayimages(noisy_images)
    pixelhistogram(images)
    colortransformations(images)
