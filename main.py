
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import os
import seaborn as sns
# %matplotlib inline
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Reshape
from torchvision import transforms  # For image transformations
from mtcnn.mtcnn import MTCNN as MTCNN
from facenet_pytorch import MTCNN as torchmtcnn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
from torchvision.ops import box_iou


'''
Data Ingestion
'''
def readdata(initpath) -> list:
    data = []
    with open(f"{initpath}/annotation_train.odgt") as file:
        for line in file:
            data.append(json.loads(line))
    return data

def readimages(data) -> list:

    images = []
    # for i in range(len(data)):
    for i in range(5):
        ID = data[i]['ID']
        paths = (f"{initpath}/Images/{ID}.JPG",
                 f"{initpath}/Images 2/{ID}.JPG",
                 f"{initpath}/Images 3/{ID}.JPG")
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
Full Body Detection Enhancements
'''
def load_detection_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model


def apply_nms(orig_prediction, iou_thresh=0.3):
    # Convert to torch tensors
    # boxes = torch.tensor(orig_prediction['boxes'])
    # scores = torch.tensor(orig_prediction['scores'])
    boxes = orig_prediction['boxes'].clone().detach()
    scores = orig_prediction['scores'].clone().detach()
    # Apply non-maximum suppression
    keep = nms(boxes, scores, iou_thresh)
    final_prediction = {
        'boxes': boxes[keep].numpy(),
        'scores': scores[keep].numpy(),
    }
    return final_prediction

def detect_full_body_nms(images, model, score_thresh=0.8, iou_thresh=0.5):
    predictions = []
    # Move the model to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():  # No need to track gradients
        for image in images:
            image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
            pred = model(image_tensor)
            # Filter out predictions below the confidence threshold
            pred_conf = pred[0]['scores'] > score_thresh
            boxes = pred[0]['boxes'][pred_conf].to('cpu')
            scores = pred[0]['scores'][pred_conf].to('cpu')
            # Apply Non-Maximum Suppression
            final_pred = apply_nms({'boxes': boxes, 'scores': scores}, iou_thresh)
            predictions.append(final_pred)
    
    return predictions

def visualize_detections(images, predictions):
    for i, img in enumerate(images):
        img_copy = img.copy()
        for box, score in zip(predictions[i]['boxes'], predictions[i]['scores']):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{score:.2f}"
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow(f'Image {i} with Full Body Detections', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


'''
Evaluation of the Detection Model
'''

def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gt in zip(predictions, ground_truths):
        # Convert ground truth boxes into a tensor
        gt_boxes = torch.tensor([box['hbox'] for box in gt])  # Ensure the key 'hbox' exists and is correct

        # Convert predicted boxes and scores into tensors
        pred_boxes = torch.tensor(pred['boxes'])
        pred_scores = torch.tensor(pred['scores'])

        # If no ground truth boxes, all detections are false positives
        if len(gt_boxes) == 0:
            false_positives += len(pred_boxes)
            continue

        # If no predictions, all ground truths are false negatives
        if len(pred_boxes) == 0:
            false_negatives += len(gt_boxes)
            continue

        # Calculate IoU for each predicted box with ground truth boxes
        ious = box_iou(gt_boxes, pred_boxes)

        # For each ground truth, find the prediction with the highest IoU
        iou_max, iou_max_index = ious.max(dim=1)

        for idx, (iou, pred_idx) in enumerate(zip(iou_max, iou_max_index)):
            if iou >= iou_threshold and pred_scores[pred_idx] > 0:
                true_positives += 1
                # Mark this prediction as used
                pred_scores[pred_idx] = -1
            else:
                false_negatives += 1

        # Remaining predictions are false positives
        false_positives += (pred_scores > 0).sum().item()

    # Compute precision, recall, and F1 score
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # initpath = ".\\CrowdHuman_Dataset\\"
    initpath = "/Users/odibia/Documents/AdvancedMachineLearning/PROJECT/CrowdHuman_Dataset"
    data = readdata(initpath)
    images = readimages(data)
    #displayimages(images)
    #noisy_images = augment_images_with_noise(images)
    #displayimages(noisy_images)  # Display augmented images
    #colortransformations(images)
    #pixelhistogram(images)
    testdetection(data, images)
    image_size_vs_objects(images, data)
    # Load the detection model
    detection_model = load_detection_model()
    # Perform full body detection
    body_detections_nms = detect_full_body_nms(images, detection_model)
    ground_truth_boxes = [entry['gtboxes'] for entry in data[:len(images)]]
    # Evaluate the detections
    evaluation_results = calculate_metrics(body_detections_nms, ground_truth_boxes)
    print(evaluation_results)
    # Visualize detections
    visualize_detections(images, body_detections_nms)
