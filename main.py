# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
Image Rotation
'''


def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def add_rotate_images(images, angle):
    rotated_images = [rotate_image(image, angle) for image in images]
    return rotated_images


'''
Image mirror
'''

def mirror_images(images):
    mirrored_images = [cv2.flip(image,1) for image in images] # Flip images horizontally (along the y-axis)
    return mirrored_images



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

'''
Image Size vs. Objects Detected
'''
def image_size_vs_objects(images, data) -> None:
    sizes = [img.shape[0] * img.shape[1] for img in images]
    objects = [len(entry['gtboxes']) for entry in data[:len(images)]]

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, objects, color='blue')
    plt.title('Image Sizes vs. Number of Objects Detected')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.grid(True)
    plt.show()

    # Correlation Coefficient
    correlation_matrix = np.corrcoef(sizes, objects)
    correlation_coefficient = correlation_matrix[0, 1]

    # Correlation Coefficient Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, objects, color='red')
    plt.title(f'Image Sizes vs. Number of Objects Detected\nCorrelation Coefficient: {correlation_coefficient:.3f}')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.grid(True)
    plt.show()

    # Data for Violin Plot
    df = pd.DataFrame({
        'Image Size': sizes,
        'Number of Objects Detected': objects
    })

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Image Size', y='Number of Objects Detected', data=df)
    plt.title('Distribution of Number of Objects Detected Across Image Sizes')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Heatmap
    plt.figure(figsize=(10, 6))
    plt.hexbin(sizes, objects, gridsize=30, mincnt=1)
    plt.colorbar(label='Count in bin')
    plt.title('Density of Objects Detected vs. Image Size')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.grid(True)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initpath = ".\\CrowdHuman_Dataset\\"
    data = readdata(initpath)
    images = readimages(data)
    displayimages(images)
    noisy_images = augment_images_with_noise(images)
    displayimages(noisy_images)  # Display augmented images
    colortransformations(images)
    pixelhistogram(images)
    image_size_vs_objects(images, data)


#######  RetinaFace Pytorch

import cv2
from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
plt.rcParams["figure.figsize"] = (15, 15)
# imagecv2 = cv2.imread('1.jpg')
imagecv2 = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

import matplotlib.pyplot as plt
# Assuming imagecv2 is your list of arrays
for img_array in imagecv2:
    plt.imshow(img_array)
    plt.show()
displayimages(imagecv2)
model = get_model("resnet50_2020-07-20", max_size=2048)

# model.eval()
# annotation = [model.predict_jsons(image) for image in imagecv2]
# annotation= model.predict_jsons(imagecv2)
annotations=[]
for imag in imagecv2:
    annotation = model.predict_jsons(imag)
    annotations.append(annotation)
    plt.show()
##
imagecv2 = [np.asarray(img, dtype=np.uint8) for img in imagecv2]
