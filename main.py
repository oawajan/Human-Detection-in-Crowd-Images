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
    for i in range(10):
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
Input Image Exploration
'''


def displayimages(images, count, initpath) -> None:

    for i in range(count):
        cv2.imshow(str(i), images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def pixelhistogram(count, df)->None:

    for i in range(count):
        ID = df[i]['ID']
        print(df[i])
        paths = (f"{initpath}CrowdHuman_train01\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train02\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train03\\Images\\{ID}.JPG")
        for path in paths:
            img = cv2.imread(path)
            if (img is not None):
                vals = img.mean(axis=2).flatten()

                counts,bins = np.histogram(vals,range(257))
                plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
                plt.xlim([-0.5, 255.5])
                plt.xlabel('Pixel Intensity')
                plt.ylabel('Frequency')
                plt.title(f'Pixel Intensity Histogram {ID}')
                plt.show()


'''
Data Augmentation
'''

def colortransformations(count, df)->None:

    for i in range(count):
        ID = df[i]['ID']
        print(df[i])
        paths = (f"{initpath}CrowdHuman_train01\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train02\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train03\\Images\\{ID}.JPG")
        for path in paths:
            img = cv2.imread(path)
            if (img is not None):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                cv2.imshow(ID, hsv)
                cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initpath = "C:\\Users\\omara\\OneDrive - University of Vermont\\CrowdHuman_Dataset\\"
    data = readdata(initpath)
    images = readimages(data)
    displayimages(images, 5, initpath)
    # pixelhistogram(5, data)
    # colortransformations(5, data)