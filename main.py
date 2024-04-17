import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
import torch
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

'''
Data Ingestion
'''
# Function to read .odgt data file line by line and parse JSON objects
def readdata(initpath, filename) -> list:
    data = []
    filepath = os.path.join(initpath, filename)
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Parse each line separately and append to list
    return data

# Function to load images based on annotation data
def readimages(data, initpath) -> list:
    images = []
    for i in range(min(5, len(data))):  # Process a maximum of 5 images for demonstration
        ID = data[i]['ID']
        img = None
        for folder in ["Images", "Images 2", "Images 3"]:
            img_path = os.path.join(initpath, folder, f"{ID}.JPG")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                break
        if img is not None:
            images.append(img)
        else:
            print(f"Image {ID} not found in any directory.")
    return images

'''
Gaussian Noise Addition
'''
# Adds Gaussian noise to an image
def add_gaussian_noise(image, mean=0, sigma=25):
    height, width, _ = image.shape  # Get dimensions of the image
    noise = np.random.normal(mean, sigma, (height, width, 3))  # Generate Gaussian noise
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)  # Add noise and clip the image pixel values
    return noisy_image

# Applies Gaussian noise to each image in a list
def augment_images_with_noise(images):
    return [add_gaussian_noise(image) for image in images]

'''
Input Image Exploration
'''
# Displays each image in a list
def displayimages(images) -> None:
    for i in range(len(images)):
        cv2.imshow(str(i), images[i])  # Display image in a window
        cv2.waitKey(0)  # Wait for a key press to proceed
        cv2.destroyAllWindows()  # Close the window

# Generates and displays a histogram of pixel intensities for each image
def pixelhistogram(images) -> None:
    for img in images:
        vals = img.mean(axis=2).flatten()  # Compute the mean of pixel intensities
        counts, bins = np.histogram(vals, range(257))  # Generate histogram data
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')  # Create a bar plot
        plt.xlim([-0.5, 255.5])  # Set x-axis limits
        plt.xlabel('Pixel Intensity')  # Set x-axis label
        plt.ylabel('Frequency')  # Set y-axis label
        plt.title('Pixel Intensity Histogram')  # Set title
        plt.show()  # Display the plot

'''
Data Augmentation
'''
# Applies and displays a color transformation (BGR to HSV) for each image
def colortransformations(images) -> None:
    for i, img in enumerate(images):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert BGR image to HSV format
        cv2.imshow(str(i), hsv)  # Display the transformed image
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the display window

'''
Image Size vs. Objects Detected
'''
# Analyzes the relationship between image sizes and the number of objects detected
def image_size_vs_objects(images, data) -> None:
    sizes = [img.shape[0] * img.shape[1] for img in images]  # Calculate the size of each image
    objects = [len(entry['gtboxes']) for entry in data[:len(images)]]  # Count objects detected in each image

    # Generate various plots to visualize the relationship between image size and objects detected
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, objects, color='blue')
    plt.title('Image Sizes vs. Number of Objects Detected')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.grid(True)
    plt.show()

    correlation_matrix = np.corrcoef(sizes, objects)
    correlation_coefficient = correlation_matrix[0, 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, objects, color='red')
    plt.title(f'Correlation Coefficient: {correlation_coefficient:.3f}')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.grid(True)
    plt.show()

    df = pd.DataFrame({'Image Size': sizes, 'Number of Objects Detected': objects})
    sns.violinplot(x='Image Size', y='Number of Objects Detected', data=df)
    plt.title('Distribution Across Image Sizes')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hexbin(sizes, objects, gridsize=30, cmap='Blues')
    plt.colorbar(label='Count in bin')
    plt.title('Density vs. Image Size')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.grid(True)
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, root, annotations_file, transform=None):
        self.root = root
        with open(os.path.join(root, annotations_file), 'r') as f:
            self.annotations = [json.loads(line.strip()) for line in f]
        self.transform = transform

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['ID'].replace(',', '') + '.JPG'
        img_path = os.path.join(self.root, img_id)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Image {img_id} not found or cannot be opened.")
            return None  # Return None if image cannot be loaded

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        for box in annotation['gtboxes']:
            if box['tag'] == 'person':
                xmin = box['fbox'][0]
                ymin = box['fbox'][1]
                xmax = xmin + box['fbox'][2]
                ymax = ymin + box['fbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        targets = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.annotations)

def get_data_loaders(root, annotations_file, transform, batch_size=2):
    dataset = CustomDataset(root, annotations_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Filter out None values
    return torch.utils.data.dataloader.default_collate(batch)

def create_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = RetinaNet(backbone, num_classes=num_classes)
    model.to(device)
    return model

def train_model(model, data_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}')

def main():
    train_dir = './CrowdHuman_Dataset'
    train_annotation = 'annotation_train.odgt'
    num_classes = 2  # Background and person

    data_loader = get_data_loaders(train_dir, train_annotation, transform=F.to_tensor)
    model = create_model(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    train_model(model, data_loader, optimizer, 10)

if __name__ == '__main__':
    initpath = "./CrowdHuman_Dataset"  # Define the path to the dataset
    data = readdata(initpath, "annotation_train.odgt")  # Read data from the annotation file
    images = readimages(data, initpath)  # Load images based on the annotations
    displayimages(images)  # Display the images
    noisy_images = augment_images_with_noise(images)  # Apply noise to images
    displayimages(noisy_images)  # Display noisy images
    colortransformations(images)  # Apply and display color transformations
    pixelhistogram(images)  # Display pixel histograms
    image_size_vs_objects(images, data)  # Analyze image size vs objects detected
    main()
