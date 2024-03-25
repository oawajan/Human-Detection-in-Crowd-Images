# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
from mtcnn.mtcnn import MTCNN as MTCNN
from facenet_pytorch import MTCNN as torchmtcnn
import os


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

'''
Detection in PreTrained models
'''

def detectopencv(images) -> list:
    predictions = []
    for image in images:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
        predictions.append(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Opencv2 Face Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return predictions


def detectMTCNN(images) -> list:
    predictions = []
    model = MTCNN(min_face_size=20, scale_factor=0.709)
    for image in images:
        faces = model.detect_faces(image)
        facelist = []
        for face in faces:
            x, y, w, h = face['box']
            facelist.append(face['box'])
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        predictions.append(facelist)
        cv2.imshow('MTCNN Face Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return predictions


def detecttorch(images) -> list:
    model = torchmtcnn()
    predictions = []
    for image in images:
        faces, probs, landmarks = model.detect(image, landmarks=True)
        faces_list = []
        predictions.append(faces)
        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)
        cv2.imshow('Pytorch MTCNN Face Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return predictions


def IoU(data, predictions) -> list:
    i = 0
    results = []
    for prediction in predictions:
        temp = []
        for boxes in prediction:
            max_IoU = 0
            for box in range(len(data[i]['gtboxes'])):
                x_intersection = max(int(boxes[0]),
                                     data[i]['gtboxes'][box]['hbox'][0])

                y_intersection = max(int(boxes[1]),
                                     data[i]['gtboxes'][box]['hbox'][1])

                w_intersection = min(int(boxes[0]) + int(boxes[2]),
                                     data[i]['gtboxes'][box]['hbox'][0] + data[i]['gtboxes'][box]['hbox'][
                                         2]) - x_intersection

                h_intersection = min(int(boxes[1]) + int(boxes[3]),
                                     data[i]['gtboxes'][box]['hbox'][1] + data[i]['gtboxes'][box]['hbox'][
                                         3]) - y_intersection
                area_intersection = max(0, w_intersection) * max(0, h_intersection)

                area_union = (int(boxes[2]) * int(boxes[3])
                              + data[i]['gtboxes'][box]['hbox'][2] * data[i]['gtboxes'][box]['hbox'][3]
                              - area_intersection)
                IoU = area_intersection / area_union
                max_IoU = max(max_IoU, IoU)

            temp.append(max_IoU)
            i += 1
        results.append(temp)
    return results


def testdetection(data,images) -> None:
    results = []
    IoUresults = []

    # results.append(detectopencv(images))
    # results.append(detectMTCNN(images))
    # results.append(detecttorch(images))

    for result in results:
        IoUresults.append(IoU(data, result))

    for result in IoUresults:
        print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    initpath = ".\\CrowdHuman_Dataset\\"
    data = readdata(initpath)
    images = readimages(data)
    #displayimages(images)
    #noisy_images = augment_images_with_noise(images)
    #displayimages(noisy_images)  # Display augmented images
    #colortransformations(images)
    #pixelhistogram(images)
    #testdetection(data, images)
    image_size_vs_objects(images, data)
