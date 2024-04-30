import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import os
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as nnF  # Alias for torch.nn.functional to avoid conflict with torchvision.transforms.functional
from torchvision.transforms import functional as TVF  # Alias for torchvision.transforms.functional
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
from torchvision.ops import box_iou

# Data Ingestion
def readdata(initpath) -> list:
    """Reads JSON formatted data from a specified path."""
    data = []
    with open(f"{initpath}/annotation_train.odgt") as file:
        for line in file:
            data.append(json.loads(line))
    return data

def readimages(data) -> list:
    """Loads images based on the IDs found in the data, from three different directories."""
    images = []
    for i in range(5):  # Limit to first 5 entries for example
        ID = data[i]['ID']
        paths = (f"{initpath}/Images/{ID}.JPG",
                 f"{initpath}/Images 2/{ID}.JPG",
                 f"{initpath}/Images 3/{ID}.JPG")
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
    return images

# Gaussian Noise Addition
def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    height, width, _ = image.shape
    noise = np.random.normal(mean, sigma, (height, width, 3))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def augment_images_with_noise(images):
    """Applies Gaussian noise to a list of images."""
    return [add_gaussian_noise(image) for image in images]

# Input Image Exploration
def displayimages(images) -> None:
    """Displays images one at a time."""
    for i, image in enumerate(images):
        cv2.imshow(f'Image {i}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def pixelhistogram(images) -> None:
    """Displays a histogram of pixel intensities for each image."""
    for img in images:
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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow(f'HSV Image {i}', hsv)
        cv2.waitKey(0)

# Image Size vs. Objects Detected
def image_size_vs_objects(images, data) -> None:
    """Analyzes the relationship between image size and the number of objects detected."""
    sizes = [img.shape[0] * img.shape[1] for img in images]
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

# Full Body Detection Enhancements
def load_detection_model():
    """Loads a pre-trained Faster R-CNN model with ResNet50 backbone."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model

def apply_nms(orig_prediction, iou_thresh=0.3):
    """Applies Non-Maximum Suppression to filter predictions based on IoU threshold."""
    boxes = orig_prediction['boxes'].clone().detach()
    scores = orig_prediction['scores'].clone().detach()
    keep = nms(boxes, scores, iou_thresh)
    final_prediction = {
        'boxes': boxes[keep].numpy(),
        'scores': scores[keep].numpy(),
    }
    return final_prediction

def detect_full_body_nms(images, model, score_thresh=0.8, iou_thresh=0.5):
    """Detects full body using a pre-trained model and applies NMS."""
    predictions = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():  # Disable gradient tracking
        for image in images:
            image_tensor = TVF.to_tensor(image).unsqueeze(0).to(device)
            pred = model(image_tensor)
            pred_conf = pred[0]['scores'] > score_thresh
            boxes = pred[0]['boxes'][pred_conf].to('cpu')
            scores = pred[0]['scores'][pred_conf].to('cpu')
            final_pred = apply_nms({'boxes': boxes, 'scores': scores}, iou_thresh)
            predictions.append(final_pred)
    return predictions

def visualize_detections(images, predictions):
    """Visualizes the detection results on the images."""
    for i, img in enumerate(images):
        img_copy = img.copy()
        for box, score in zip(predictions[i]['boxes'], predictions[i]['scores']):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{score:.2f}"
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow(f'Image {i} with Full Body Detections', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Evaluation of the Detection Model
def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    """Calculates precision, recall, and F1-score for the detection results."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gt in zip(predictions, ground_truths):
        gt_boxes = torch.tensor([box['hbox'] for box in gt])
        pred_boxes = torch.tensor(pred['boxes'])
        pred_scores = torch.tensor(pred['scores'])

        if len(gt_boxes) == 0:
            false_positives += len(pred_boxes)
            continue

        if len(pred_boxes) == 0:
            false_negatives += len(gt_boxes)
            continue

        ious = box_iou(gt_boxes, pred_boxes)
        iou_max, iou_max_index = ious.max(dim=1)

        for idx, (iou, pred_idx) in enumerate(zip(iou_max, iou_max_index)):
            if iou >= iou_threshold and pred_scores[pred_idx] > 0:
                true_positives += 1
                pred_scores[pred_idx] = -1  # Mark this prediction as used
            else:
                false_negatives += 1

        false_positives += (pred_scores > 0).sum().item()

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

# Multilayer Perceptron Neural Network
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

    def forward(self, x):
        return self.layers(x)

# Initialize and Integrate MLP into the pipeline
def initialize_mlp(input_dim, hidden_dim, output_dim):
    """Initializes the MLP model with specified dimensions."""
    mlp_model = MLP(input_dim, hidden_dim, output_dim)
    return mlp_model

# Main Execution Block
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Environment variable to allow multiple OpenMP libraries
    initpath = "/Users/Waza3ii/PycharmProjects/Human-Detection-in-Crowd-Images/CrowdHuman_Dataset"
    data = readdata(initpath)
    images = readimages(data)
    displayimages(images)
    noisy_images = augment_images_with_noise(images)
    displayimages(noisy_images)  # Display augmented images
    colortransformations(images)
    pixelhistogram(images)
    image_size_vs_objects(images, data)
    detection_model = load_detection_model()
    body_detections_nms = detect_full_body_nms(images, detection_model)
    ground_truth_boxes = [entry['gtboxes'] for entry in data[:len(images)]]
    evaluation_results = calculate_metrics(body_detections_nms, ground_truth_boxes)
    visualize_detections(images, body_detections_nms)
    input_dim = 2048  # Assumed input feature size
    hidden_dim = 512  # Hidden layer size
    output_dim = 10  # Number of classes
    mlp_model = initialize_mlp(input_dim, hidden_dim, output_dim)
    mlp_model.eval()
    dummy_input = torch.randn(1, input_dim)
    output = mlp_model(dummy_input)
    probabilities = nnF.softmax(output, dim=1)  # Convert logits to probabilities
    print(probabilities)
