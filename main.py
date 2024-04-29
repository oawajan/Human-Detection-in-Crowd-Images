import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.ssd import ssd300_vgg16
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

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
    print("Starting to load images...")
    for i in range(min(5, len(data))):  # Limiting to first 5 for demonstration
        ID = data[i]['ID']
        found = False
        for folder in ["Images", "Images 2", "Images 3"]:
            img_path = os.path.join(initpath, folder, f"{ID}.JPG")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    print(f"Loaded image from {img_path}")
                    found = True
                    break
        if not found:
            print(f"Failed to find image for ID {ID} in expected folders.")
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

class HumanDetectionDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None, max_boxes=50, limit=5):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = self.load_annotations(annotation_file)
        self.max_boxes = max_boxes  # Set a max number of boxes to handle variable number per image

    def load_annotations(self, file_path):
        annotations = []
        with open(file_path, 'r') as file:
            for line in file:
                annotations.append(json.loads(line.strip()))
        return annotations
    def __len__(self):
        # Ensure this method returns an integer representing the number of items in the dataset
        return len(self.annotations)
    def __getitem__(self, idx):
        img_id = self.annotations[idx]['ID']
        img_path = self.find_image_path(img_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        for box in self.annotations[idx]['gtboxes']:
            if box['tag'] == 'person':
                x, y, w, h = box['fbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(1)  # Assuming 1 is the label for 'person'

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Pad boxes to ensure consistent size across all data
        if len(boxes) < self.max_boxes:
            padded_boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - len(boxes)), "constant", 0)
        else:
            padded_boxes = boxes[:self.max_boxes]

        padded_labels = F.pad(labels, (0, self.max_boxes - len(labels)), "constant", -1)

        target = {'boxes': padded_boxes, 'labels': padded_labels}
        if self.transform:
            image = self.transform(image)

        print(f"Return types - Image: {type(image)}, Target: {type(target)}")
        return image, target

    def find_image_path(self, img_id):
        for folder in ["Images", "Images 2", "Images 3"]:
            img_path = os.path.join(self.root_dir, folder, f"{img_id}.jpg")
            if os.path.exists(img_path):
                return img_path
        raise FileNotFoundError(f"No image found for ID {img_id} in any expected folder.")

def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Consolidate images normally
    images = default_collate(images)

    # Ensure targets are correctly structured as dictionaries of tensors
    out_targets = {}
    for key in targets[0]:
        out_targets[key] = torch.stack([t[key] for t in targets])
    return images, out_targets

if __name__ == '__main__':
    initpath = "./CrowdHuman_Dataset"
    data = readdata(initpath, "annotation_train.odgt")
    images = readimages(data, initpath)
    #displayimages(images)  # Display the images
    #noisy_images = augment_images_with_noise(images)  # Apply noise to images
    #displayimages(noisy_images)  # Display noisy images
    #colortransformations(images)  # Apply and display color transformations
    #pixelhistogram(images)  # Display pixel histograms
    #image_size_vs_objects(images, data)  # Analyze image size vs objects detected

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ssd_model = ssd300_vgg16(pretrained=True)
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    faster_rcnn_model = FasterRCNN(backbone=backbone, num_classes=2, rpn_anchor_generator=anchor_generator)

    dataset = HumanDetectionDataset(
        annotation_file='/Users/Waza3ii/PycharmProjects/Human-Detection-in-Crowd-Images/CrowdHuman_Dataset/annotation_train.odgt',
        root_dir='/Users/Waza3ii/PycharmProjects/Human-Detection-in-Crowd-Images/CrowdHuman_Dataset',
        transform=transform,
        limit=5  # Limit the dataset to the first 5 images
    )

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faster_rcnn_model.to(device)
    ssd_model.to(device)

    num_epochs = 10
    learning_rate = 0.005
    optimizer = optim.SGD(list(faster_rcnn_model.parameters()) + list(ssd_model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.0005)


    def train(data_loader, optimizer, num_epochs):
        faster_rcnn_model.train()
        ssd_model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                prepared_targets = []
                for target in targets:
                    try:
                        prepared_target = {k: v.to(device) for k, v in target.items()}
                        prepared_targets.append(prepared_target)
                    except AttributeError as e:
                        print("Error in targets formatting:", e)
                        print("Current target data that caused error:", target)  # Corrected to print the actual data causing issues
                        continue  # Skip this batch or handle error

                if not prepared_targets:
                    print("No valid targets could be prepared for this batch.")
                    continue

                optimizer.zero_grad()
                try:
                    loss_dict_frcnn = faster_rcnn_model(images, prepared_targets)
                    loss_dict_ssd = ssd_model(images, prepared_targets)
                    losses = sum(loss for loss in loss_dict_frcnn.values()) + sum(loss for loss in loss_dict_ssd.values())
                    losses.backward()
                    optimizer.step()

                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch Loss: {losses.item()}")
                    running_loss += losses.item()
                except Exception as e:
                    print("Error during model forward/backward pass:", e)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}")

    train(data_loader, optimizer, num_epochs)
