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

import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

'''
Model Definition
'''
def get_model(num_classes):
    model = retinanet_resnet50_fpn(pretrained=True)
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FastRCNNPredictor(in_features * num_anchors, num_classes)
    return model

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class HumanDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations[idx]['ID']
        img_path = os.path.join(self.root_dir, f"{img_id}.JPG")
        image = Image.open(img_path).convert("RGB")
        boxes = torch.as_tensor([[0, 0, 100, 100]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        target = {"boxes": boxes, "labels": labels}
        print(f"Fetching item {idx} from dataset")  # Diagnostic print
        return image, target

from torch.optim import SGD
from torch.utils.data import DataLoader

'''
Training Loop
'''
def train_model():
    # Assuming your annotations and image paths are correctly set up
    dataset = HumanDataset(annotations=data, root_dir='./CrowdHuman_Dataset/Images/', transform=data_transforms)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10

    print("Starting model training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}...")
        for images, targets in data_loader:
            print(f"  Processing a batch...")
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: Compute predicted outputs by passing inputs to the model
            loss_dict = model(images, targets)

            # Calculate the batch loss
            losses = sum(loss for loss in loss_dict.values())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
        print(f"Epoch {epoch + 1} Loss: {running_loss / len(data_loader)}")

if __name__ == '__main__':
    initpath = ".\\CrowdHuman_Dataset\\"  # Adjusted for Windows file path
    data = readdata(initpath)
    images = readimages(data, initpath)
    # Uncomment the following lines if you wish to see the images/augmentations
    # displayimages(images)
    # noisy_images = augment_images_with_noise(images)
    # displayimages(noisy_images)
    # colortransformations(images)
    train_model()
