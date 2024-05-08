import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import Resize

# Data Ingestion
def readdata(initpath) -> list:
    """Reads JSON formatted data from a specified path."""
    data = []
    with open(f"{initpath}/annotation_train.odgt") as file:
        for line in file:
            data.append(json.loads(line))
    return data

def readimages(data, initpath) -> list:
    """Loads images based on the IDs found in the data, from three different directories."""
    images = []
    for i in range(5):  # Limit to first 5 entries for example
        ID = data[i]['ID']
        paths = (f"{initpath}/Images/{ID}.JPG",
                 f"{initpath}/Images 2/{ID}.JPG",
                 f"{initpath}/Images 3/{ID}.JPG")
        for path in paths:
            if os.path.exists(path):  # Fixed the variable name from img to path
                img = read_image(path)
                img = Resize((224, 224))(img)
                img = img.float()  # Convert input tensor to float32
                img = img.unsqueeze(0)
                images.append(img)
    return images

# Gaussian Noise Addition
def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    # Convert tensor back to numpy array for processing (assuming image is a tensor)
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    height, width, _ = image.shape
    noise = np.random.normal(mean, sigma, (height, width, 3))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    # Convert back to tensor
    noisy_image = torch.from_numpy(noisy_image).permute(2, 0, 1).unsqueeze(0)
    return noisy_image

def augment_images_with_noise(images):
    """Applies Gaussian noise to a list of images."""
    return [add_gaussian_noise(image) for image in images]

# Input Image Exploration
def displayimages(images) -> None:
    """Displays images one at a time."""
    for i, image in enumerate(images):
        # Convert tensor to numpy array for display
        image = image.squeeze(0).permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.title(f'Image {i}')
        plt.show()

def pixelhistogram(images) -> None:
    """Displays a histogram of pixel intensities for each image."""
    for img in images:
        # Convert tensor to numpy array for histogram calculation
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        vals = img.mean(axis=2).flatten()
        counts, bins = np.histogram(vals, range(257))
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        plt.xlim([-0.5, 255.5])
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Pixel Intensity Histogram')
        plt.show()

# Data Augmentation
def colortransformations(images) -> None:
    """Converts images to HSV color space and displays them."""
    for i, img in enumerate(images):
        # No need to convert to OpenCV format if already tensors
        hsv = transforms.ColorJitter(hue=0.1, saturation=0.1, brightness=0.1)(img)
        plt.imshow(hsv.squeeze(0).permute(1, 2, 0).numpy())
        plt.title(f'HSV Image {i}')
        plt.show()

# Image Size vs. Objects Detected
def image_size_vs_objects(images, data) -> None:
    """Analyzes the relationship between image size and the number of objects detected."""
    sizes = [img.shape[1] * img.shape[2] for img in images]  # Assuming tensor shape is (channels, height, width)
    objects = [len(entry['gtboxes']) for entry in data[:len(images)]]

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

    df = pd.DataFrame({
        'Image Size': sizes,
        'Number of Objects Detected': objects
    })
    sns.violinplot(x='Image Size', y='Number of Objects Detected', data=df)
    plt.title('Distribution Across Image Sizes')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hexbin(sizes, objects, gridsize=30, mincnt=1)
    plt.colorbar(label='Count in bin')
    plt.title('Density vs. Image Size')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Objects Detected')
    plt.grid(True)
    plt.show()

# Full Body Detection with MLP
class MLP(nn.Module):
    """Defines a simple Multilayer Perceptron with three linear layers and ReLU activations."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Explicitly set the data type of the linear layers' weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.float()  # Convert weights to float

    def forward(self, x):
        return self.layers(x)

def initialize_mlp(input_dim, hidden_dim, output_dim):
    """Initializes the MLP model with specified dimensions."""
    mlp_model = MLP(input_dim, hidden_dim, output_dim)
    return mlp_model

def detect_full_body_mlp(images, model, threshold=0.5):
    """Detects full human bodies using the MLP model."""
    predictions = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for image in images:
            # No need to convert image to tensor, as it's already a tensor
            pred = model(image)
            # Assuming output is (x1, y1, x2, y2)
            pred_boxes = pred[0]
            # Apply threshold for detection
            if pred_boxes[0].item() > threshold:  # Extract the item and compare
                predictions.append(pred_boxes.cpu().numpy())
            else:
                predictions.append(None)
    return predictions

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    # Convert boxes to format (x1, y1, x2, y2)
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If the intersection is non-positive (no intersection), return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou

def evaluate_mlp_performance(predictions, ground_truth_boxes, threshold=0.5):
    """Evaluates the performance of the MLP model."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gt in zip(predictions, ground_truth_boxes):
        if pred is None:
            if len(gt) == 0:
                continue
            else:
                false_negatives += 1
                continue

        iou = calculate_iou(pred, gt)
        if iou >= threshold:
            true_positives += 1
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    initpath = "/Users/Waza3ii/PycharmProjects/Human-Detection-in-Crowd-Images/CrowdHuman_Dataset"
    data = readdata(initpath)
    images = readimages(data, initpath)
    #displayimages(images)
    #noisy_images = augment_images_with_noise(images)
    #displayimages(noisy_images)
    #colortransformations(images)
    #pixelhistogram(images)
    #image_size_vs_objects(images, data)

    hidden_dim = 512
    output_dim = 4

    # Update input dimension based on the processed image size
    input_dim = 3 * 224 * 224  # Assuming the images are resized to (224, 224)

    # Initialize MLP model
    mlp_model = initialize_mlp(input_dim, hidden_dim, output_dim)
    mlp_model.eval()

    # Extract ground truth boxes
    ground_truth_boxes = [entry['gtboxes'] for entry in data[:len(images)]]

    # Detect full body using MLP
    body_detections_mlp = detect_full_body_mlp(images, mlp_model, ground_truth_boxes)

    # Evaluate MLP performance
    evaluation_results_mlp = evaluate_mlp_performance(body_detections_mlp, ground_truth_boxes)

    print("MLP Performance Metrics:")
    print(evaluation_results_mlp)
