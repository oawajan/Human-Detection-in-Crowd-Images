# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
import torch.nn as nn
from facenet_pytorch import MTCNN as torchmtcnn
from mtcnn.mtcnn import MTCNN as MTCNN
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.utils.data.dataloader import default_collate


'''
Data Ingestion
'''


def collate_fn(batch):
    filtered_batch = []
    for item in batch:
        if item is not None:
            print(type(item))
            filtered_batch.append(item)
    return default_collate(filtered_batch)


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
Detection testing in PreTrained models
'''


def testopencv(images) -> list:
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


def testMTCNN(images) -> list:
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


def testtorch(images) -> list:
    print(f"GPU is available {torch.cuda.is_available()}")
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(torch.version.cuda)
    model = torchmtcnn(device=device)
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


'''
retraining the pytorch facenet mtcnn model
'''


class ImageDataset(Dataset):
    def __init__(self, target_transform=None):
        self.init_path = f".\\CrowdHuman_Dataset\\"
        self.image_labels = self.read_annotations()

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image = self.load_image(self.image_labels[index])
        print(self.image_labels[index]['ID'])
        print(type(image))
        print('-'*20)
        return image

    def read_annotations(self) -> list:
        datafile = []
        annotations = []
        with open(f"{self.init_path}annotation_train.odgt") as file:
            for line in file:
                datafile.append(json.loads(line))
        return datafile

    def load_image(self, image_annotation):
        ID = image_annotation['ID']
        paths = (f"{self.init_path}CrowdHuman_train01\\Images\\{ID}.JPG",
                 f"{self.init_path}CrowdHuman_train02\\Images\\{ID}.JPG",
                 f"{self.init_path}CrowdHuman_train03\\Images\\{ID}.JPG")
        try:
            for path in paths:
                img = read_image(path)
                if img is not None:
                    img = img.permute(1, 2, 0)
                    plt.imshow(img)
                    plt.title(ID)
                    plt.show()
                    print(type(img))
                    return img
        except Exception as e:
            print(f"Error reading image from: {str(e)}")


def trainMTCNN(data, images) -> list:

    if (torch.cuda.is_available()):
        print(f"GPU is available {torch.cuda.is_available()}, Version {torch.version.cuda}")
        device = torch.device('cuda')
    else:
        print(f"Using CPU")
        device = torch.device('cpu')
    # initialize the MTCNN model and set it to use the GPU
    model = torchmtcnn(device=device)
    # print(model)
    print("##################freeze pretrained model layers except  one#############################")
    # freeze pretrained model layers
    for name, param in torchmtcnn.named_parameters(model):
        # print(f"name: {name}")
        # print(f"param: {param}")
        if 'pnet.conv1' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("##################              load dataset                #############################")
    dataset = ImageDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # epoch count
    epochs = 10
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


'''
Validate model results
'''


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

    # results.append(testopencv(images))
    # results.append(testMTCNN(images))
    #results.append(testtorch(images))

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
    # displayimages(images)
    # noisy_images = augment_images_with_noise(images)
    # displayimages(noisy_images)  # Display augmented images
    # colortransformations(images)
    # pixelhistogram(images)
    # testdetection(data, images)
    # image_size_vs_objects(images, data)
    trainMTCNN(data, images)
