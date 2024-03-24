# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2


'''
Data Ingestion
'''


def readdata(initpath) -> list:
    data = []
    with open(f"{initpath}annotation_train.odgt") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def readimages(data) -> list:

    images = []
    # for i in range(len(data)):
    for i in range(5):
        ID = data[i]['ID']
        paths = (f"{initpath}CrowdHuman_train01\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train02\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train03\\Images\\{ID}.JPG")
        for path in paths:
            img = cv2.imread(path)
            if (img is not None):
                images.append(img)

    return images




'''
Gaussian Noise Addition
'''
def add_gaussian_noise(image, mean=0, sigma=25):
    height, width, _ = image.shape
    noise = np.random.normal(mean, sigma, (height, width, 3))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def augment_images_with_noise(images):
    return [add_gaussian_noise(image) for image in images]



'''
Input Image Exploration
'''


def displayimages(images) -> None:

    for i in range(len(images)):
        cv2.imshow(str(i), images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def pixelhistogram(images)->None:

    for img in images:
        vals = img.mean(axis=2).flatten()
        counts,bins = np.histogram(vals,range(257))
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        plt.xlim([-0.5, 255.5])
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Pixel Intensity Histogram')
        plt.show()


'''
Data Augmentation
'''

def colortransformations(images)->None:
    i = 0
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow(str(i), hsv)
        cv2.waitKey(0)
        i += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initpath = ".\\CrowdHuman_Dataset\\"
    data = readdata(initpath)
    images = readimages(data)
    # displayimages(images)
    noisy_images = augment_images_with_noise(images)
    displayimages(noisy_images, 5)  # Display augmented images
    pixelhistogram(images)
    colortransformations(images)
    
