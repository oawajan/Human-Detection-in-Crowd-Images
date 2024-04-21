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



###########################################################################
###########################################################################
##################################### NN MLP###############################

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Function to extract faces from annotation
def extract_faces(annotation):
    faces = []
    for gtbox in annotation['gtboxes']:
        if gtbox['tag'] == 'person':
            faces.append(gtbox['hbox'])  # Use 'hbox' for face bounding box
    return faces

# Function to prepare dataset
def prepare_dataset(annotation, image_folder, target_size=(100, 100), test_size=0.2, random_state=42):
    X = []  # Resized images
    Y = []  # Adjusted bounding box coordinates (x, y, width, height)

    for ann in annotation:
        image_id = ann['ID']
        for folder_name in ["CrowdHuman_train01", "CrowdHuman_train02", "CrowdHuman_train03"]:
            folder_path = os.path.join(image_folder, folder_name)
            if os.path.isdir(folder_path):
                image_path = os.path.join(folder_path, f"{image_id}.JPG")
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Resize image to target size
                        image_resized = cv2.resize(image, target_size)
                        # Extract bounding boxes for faces
                        faces = extract_faces(ann)
                        if faces:
                            for face in faces:
                                # Extract coordinates of the bounding box
                                x, y, w, h = face
                                # Scale bounding box coordinates to match resized image
                                x_scaled = int(x * target_size[0] / image.shape[1])
                                y_scaled = int(y * target_size[1] / image.shape[0])
                                w_scaled = int(w * target_size[0] / image.shape[1])
                                h_scaled = int(h * target_size[1] / image.shape[0])
                                # Append resized image and adjusted bounding box to the dataset
                                X.append(image_resized)
                                Y.append([x_scaled, y_scaled, w_scaled, h_scaled])
            else:
                print(f"Folder '{folder_name}' not found for image: {image_id}")

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    return (X_train, y_train), (X_test, y_test)

# Load annotations and prepare dataset
annotation_path = "C:/Users/Sepi/Downloads/ML/CrowdHuman_Dataset/annotation_train.odgt"
image_folder = "C:/Users/Sepi/Downloads/ML/CrowdHuman_Dataset"
annotations = read_annotation(annotation_path)
(X_train, y_train), (X_test, y_test) = prepare_dataset(annotations, image_folder)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Check if datasets are empty
if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
    print("Dataset is empty.")
else:
    print("Datasets prepared successfully.")


# Ctreating the model and model fitting
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_mlp_model(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(200, activation='relu'),
        # layers.Dense(200, activation='relu'),
        # layers.Dense(200, activation='relu'),
        layers.Dense(120, activation='relu'),
        layers.Dense(4)  # Output layer with 4 units for bounding box coordinates (x, y, width, height)
    ])
    return model

# Create the model
input_shape = X_train[0].shape  # Input shape is the shape of the resized images
mlp_model = create_mlp_model(input_shape)

# Compile the model
mlp_model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
mlp_model.summary()


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
# Train the model
mlp_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Testing the model on a specific image
# Reshape the test image to match the input shape expected by the model

test_image = X_test[200].reshape(1, *input_shape)

# Get the model's prediction for the test image
predicted_bbox = mlp_model.predict(test_image)

# Extract predicted bounding box coordinates and dimensions
x_pred, y_pred, w_pred, h_pred = predicted_bbox[0]

# Plot the original image
plt.imshow(X_test[200])
plt.axis('off')

# Plot the predicted bounding box
plt.gca().add_patch(plt.Rectangle((x_pred, y_pred), w_pred, h_pred, edgecolor='r', facecolor='none'))

plt.show()

