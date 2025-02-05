# CS 254 Machine Learning

# Project Proposal Report Format

## 1. Introduction

Computer vision is a crucial area of study in artificial intelligence, enabling computing systems to
understand and interpret the world around us. Computer vision, despite its progress, encounters
substantial obstacles, especially when it comes to detecting objects in crowded environments.

This study focuses on the complex problem of identifying individuals in group photographs, which is made
difficult by factors such as varying distances, sizes, body distortions, and differences within the same
category. Improving detection accuracy in these situations is not just a technological goal, but also has
significant implications for enhancing the functionality of public surveillance systems, facilitating the
indexing of people in social media images, and improving crowd management and emergency response
strategies.

This research intends to construct a model using deep learning approaches, notably Convolutional Neural
Networks (CNNs), to accurately identify human figures in complicated crowd scenarios. By utilizing
supervised learning on a dataset that has been tagged, the model will acquire the ability to accurately
identify and categorize human presence with improved precision. The Vermont Advanced Computing
Center (VACC) will serve as the computational foundation for this project, allowing for the investigation
of how different GPU architectures affect the efficiency of training and the performance of models.

By examining previous research, we place our approach within the spectrum of endeavors focused on
improving algorithms for detecting humans. Through a comparative analysis of our technique and findings
with previous studies, our objective is to emphasize the gradual improvements in both accuracy and
efficiency that our work brings to the area.

## 2. Problem Definition and Algorithm

## 2.1 Task Definition

The goal is to create an efficient and effective system for identifying human faces (and full body detection)
in photos containing many people. The system receives a wide range of group photos as inputs, which
may include multiple human faces in different positions, lighting conditions, occlusions, and densities of
persons. The intended results consist of photos that contain precisely identified and labeled human faces.
The goal is to attain excellent levels of accuracy, precision, recall, and real-time performance in face
identification, while also maintaining computational economy. This problem is significant because it has
wide-ranging applications, such as addressing social and ethical concerns related to privacy and consent
in images, improving security and surveillance for public safety, optimizing social media platforms through


automatic tagging and content filtering, advancing human-computer interaction, supporting medical and
healthcare systems, and exploring the implications for technological innovation in computer vision and
machine learning.

## 2.3. Dataset

CrowdHuman[1] is a benchmark dataset to better evaluate detectors in crowd Images. Our project utilizes
the CrowdHuman dataset, which is a comprehensive and well annotated dataset specially created for the
purpose of recognizing individuals in densely populated environments. The dataset consists of around
470,000 human examples divided into training and validation subsets. Each photograph contains an
average of 22.6 individuals and demonstrates an extensive spectrum of occlusions.

The dataset is a publicly accessible benchmark and may be downloaded via the official website, promoting
convenient access for research endeavors.

Every individual in the dataset is accurately labeled with three distinct categories of bounding boxes: head,
visible area, and entire body. The careful procedure of annotating handles many obstacles related to
occlusion and generates extensive data for algorithms that recognize humans.

Due to the dataset’s size and complex nature, it is necessary to use a setup equipped with a graphics
processing unit (GPU) in order to effectively manage the computational burden, especially when
processing data and training deep learning models.

The dataset is already split by folders to accommodate the cross-validation three way data split method,
where we train and validate different models and once we select the best performing model we will re-
train it with both training data and validation datasets then test against the test dataset.

The dataset comes with an annotation file, that contains the labels for each image, the labels are a list of
binding boundary boxes per person, called “gboxes”. Each person has 2 boxes associated with him, “hbox”
which is the label we use for face detection, “fbox” the label used for body detection.

## 2.2 Algorithm Definition

In addressing the challenge of detecting full bodies within diverse images, our baseline system leverages
a state-of-the-art object detection framework, Faster R-CNN with a ResNet50 backbone, augmented by
Multitask Cascaded Convolutional Networks (MTCNN) for enhanced face detection. This integrated
approach is designed to offer robust performance across a wide range of scenarios, including crowded
environments and varied lighting conditions. Below, we describe the algorithm in detail, justify the choice
of our baseline system, and discuss potential enhancements and alternative techniques.

## Algorithm Description (Body Detection)

**Baseline System: Faster R-CNN with ResNet50 Backbone**

- **Objective** : Detect full human bodies within images, identifying individual persons even in crowded
    settings.


- **Rationale** : Faster R-CNN is selected for its efficiency in generating high-quality region proposals
    that are then refined to accurately detect objects. The ResNet50 backbone is chosen for its depth
    and ability to learn rich feature representations, crucial for the varied aspects of human detection.

**Algorithm Workflow:**

- **Data Ingestion:**

```
Load the dataset and preprocess images (resize, normalize).
```
- **Full Body Detection (Faster R-CNN):**

```
Input: Preprocessed images.
Output: Coordinates of detected full bodies.
```
- **Post-processing:**

Apply Non-Maximum Suppression (NMS) to refine detections, removing overlaps and duplicates.
Annotate images with detection results.
**Pseudocode:**

Function detect_full_body_nms:
Input: images - List of images to process
model - Pre-trained detection model (e.g., Faster R-CNN)
score_thresh - Confidence threshold for predictions
iou_thresh - IOU threshold for Non-Maximum Suppression (NMS)
Output: predictions - List of predictions with filtered boxes and scores

Begin:

1. Initialize an empty list for predictions.
2. Determine the appropriate device (GPU/CPU) for computation.

For each image in the input list:
a. Convert the image to a tensor and move it to the computation device.
b. Use the model to predict objects in the image.
c. Filter out predictions with a confidence score lower than score_thresh.
d. Apply Non-Maximum Suppression (NMS) to the remaining predictions
using iou_thresh.
e. Store the final predictions (post-NMS) in the predictions list.

Return the list of predictions.

End


This pseudocode captures the essence of the **detect_full_body_nms** function, which processes each
image through the model, applies a confidence threshold to filter out less certain detections, and then
uses Non-Maximum Suppression to eliminate overlapping boxes, ensuring that each detected object is
represented by the most confident bounding box.

**Justification for the Baseline System**

The choice of Faster R-CNN with a ResNet50 backbone as our baseline system is motivated by the model's
proven track record in various object detection benchmarks. Its ability to learn deep, rich feature
representations makes it highly effective for the complex task of full body detection.

**Future Enhancements and Techniques**

While the current system provides a solid foundation, we plan to explore several enhancements:

- **Improved Backbone Architectures** : Investigating the use of more advanced backbone
    architectures like ResNeXt or EfficientNet for potentially better feature extraction capabilities.
- **Attention Mechanisms** : Incorporating attention mechanisms to better focus on relevant features
    within images, potentially improving detection accuracy in challenging scenarios.
- **Ensemble Methods** : Combining predictions from multiple models or variations of our baseline
    system to improve accuracy and reduce the impact of any single model's weaknesses.

**RetinaFace Pytorch model (Face Detection):**

**Objective** : A state-of-the-art solution for face detection that combines high accuracy, efficiency,
versatility, and robustness, making it suitable for a wide range of applications and deployment scenarios.

**Rationale** : to push the boundaries of face detection performance by leveraging the capabilities of deep
learning, while also ensuring efficiency, robustness, and ease of use for practical applications.

**Algorithm Workflow:**

**Data Ingestion:**

- Load the dataset and preprocess images (resize, normalize).

**Face detection (RetinaFace Pytorch):**

- Input: Preprocessed images.
- Output: Coordinates of detected


#### •

**Post Processing:**

refine and filter the raw detections through steps such as non-maximum suppression, confidence
thresholding, bounding box refinement, size filtering, and optional face alignment to improve accuracy
and prepare them for downstream tasks.

**Pseudocode:**

1. Import Required Libraries: Import the OpenCV library
2. Set Default Plot Size: Set the default figure size for matplotlib plots
3. Convert Images to RGB: Convert each image from the BGR color space to RGB and store them in
    a new list
4. Display Original Images: Iterate over each image array, display the current image using
    matplotlib's imshow function, show the plot
5. Load and Prepare RetinaFace Model: Load the RetinaFace model, set the model to evaluation
    mode
6. Predict Annotations: Initialize an empty list named, for each image use the RetinaFace model to
    predict annotations
7. Display Annotated Images: Iterate over each pair of image and annotation, apply the annotations
    to the corresponding image function, display the annotated image and show the plot.


## MTCNN Pytorch model (Face Detection):

We also implemented an MTCNN model, which is Multitask CNN model, with its architecture
made of 3 models each serving a different purpose, for steps of how this model works:

```
1 - The image is first rescaled to a pyramid of different scales, to provide the model with the ability
to detect faces on different scales for the same face and image, also known as image pyramid.
2 - Proposal Network, P-Net, which proposes a bunch of candidate facial regions.
3 - Refine Network, R-Net, which filters through the candidate facial regions to the best one.
4 - Output Network, O-Net, this final model is responsible for detecting landmarks highlighting a
facial expression within the finalized candidate box.
```
Below is a brief visual demonstration of the algorithm:

For its implementation, we opted for the pytorch facenet MTCNN model, the same model we used below
for testing against our data.

Below is the detailed model architecture.


```
Re-training MTCNN model workflow
1 - Declare the model and set it to use the GPU.
2 - Freeze all layer except for the first layer in the first model.
Pnet.conv
3 - Ingest the dataset into the custom data set class.
4 - Use the dataset loader to create a set of training batches, we have implemented a custom collate
function “collate_fn()” as we have some inconsistent data in our dataset, the function returns to
us a concatenated batch of properly read images.
5 - We are currently working on the optimizer function.
6 - We will feed our data through a training loop.
```
## 3. Experimental Evaluation

3.1 Methodology
After ingesting the dataset, we started visualizing the dataset and images and did that over 2 major steps:

```
1 - Data transformation:
a. Read and displayed a set of images using opencv2 library. This was done to verify that
we can read and handle image data properly.
```

b. Transformed the images coloring space from BGR to HSV and other spaces available,
that was done to see how regions of interest show in different color spaces.

c. different degrees of gaussian noise added , which is a common method to solve
overfitting our datasets as at increases the variance in our datasets.


```
d. Rotating introduces variations in orientation and pose of faces. This makes the model
more robust to different orientations and poses that may be encountered in real-world
scenarios. For example, a face detection model trained on rotated images is more likely
to accurately detect faces even if they are tilted or facing different directions.
```
```
e. By applying mirrored transformation to the images, we effectively create new variations
of the original data. This helps in preventing overfitting and improves the generalization
capability of the model.
```
2. Data visualization


```
f. Pixel intensity histograms: plotting per images the frequency, as this also provides us
with insight with distribution of pixel colors.
```
Then we have utilized pretrained models against our data, and that have helped us to view how
different benchmark implementations and architectures performs with our dataset.

The 4 pretrained models are

```
1 - Opencv2 cascade classifier, we can see that the model does have a weaker detection capability.
```
```
2 - MTCNN[4] by Kaipeng Zhang and it does offer a stronger face detection capability.
```

```
3 - Pytorch MTCNN[^5 ] does offer a stronger face detection capability.
```
```
4 - RetinaFace Pytorch : employs a multi-task deep learning architecture capable of detecting faces,
estimating their 3D pose, and predicting facial landmarks simultaneously, offering
comprehensive face analysis functionalities in a single model.
```
We have then implemented an evaluation method called union over intersection, which offers an
intersect accuracy percentage of the detected box and ground truth box, this is usually done by dividing
the intersection of the boxes over the union, per the following equations.


In the case of multiple boxes within the image, the case of our dataset we execute this calculation for each
predicted box against all ground truth boxes and we will pick the largest match.

**Body Detection:**

Evaluation Criteria/Performance Data/ Hypotheses

Our proposed method's evaluation is grounded on widely accepted metrics in the field of object detection,
emphasizing precision, recall, F1 score, etc across different Intersection over Union (IoU) thresholds.
These metrics collectively offer a balanced view of the model's accuracy, reliability, and robustness. For
performance data for the body detection, we collected {'true_positives': 0, 'false_positives': 65,
'false_negatives': 100, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0}. However, while our model detects full
bodies in images, our model is still underperforming and further debugging and hyperparameter tuning
will be carried out to optimize performance and accuracy.

We hypothesize that our enhanced detection system, leveraging a combination of Faster R-CNN with a
ResNet50 backbone for full body detection, provides superior accuracy and efficiency in detecting
individuals in varied and challenging environments compared to standard detection models.

## 3.2 Results

The figure below offers a visualization that depicts the link between the sizes of the images (x-axis) and
the number of items discovered inside these images (y-axis).

Each point on the scatter plot corresponds to a specific image from the dataset. The horizontal location
of the point represents the size of the image in pixels, while the vertical position denotes the number of
objects detected in that image. The concentration of points at the lower left end of the image size scale
indicates that most photos in the dataset are of smaller dimensions. The range of items detected
indicates the diversity in object counts, which does not appear to have a strong correlation with image
size, as evidenced by the distribution of data points across different sizes.

The figure below can facilitate the development of a streamlined and productive system for object
identification. It can help determine if the existing face detection system performs equally well across
different image sizes or if there are specific image size ranges where the system's performance differs
from the expected number of detections. If larger photos continually exhibit a lower number of
detections than anticipated, it may suggest the need to enhance the detection procedures. If smaller
images yield a higher number of detections compared to larger ones, it implies that the algorithm is
responsive to image resolution. In such cases, it may be advantageous to use pre-processing techniques
such scaling the images to a uniform scale.

Gaining a thorough understanding of this connection is crucial in order to accomplish the project's goals


of obtaining high levels of accuracy and computational efficiency, especially when dealing with a wide
range of inputs such as group images taken under different densities and situations. The information
obtained from this plot might be used to improve the creation of preprocessing routines, optimize
detection algorithms, and balance computational load. These improvements would greatly contribute to
the project's overall objective.

The figure below is a scatter enhanced with the correlation coefficient, quantitatively delineates the
relationship between image sizes and the number of objects detected within those images.

This plot, like the figure previously, incorporates the correlation coefficient, which stands at -0.049. Such
a value, being near zero, indicates a lack of significant linear relationship between the size of an image
and the number of objects detected within it, a conclusion visually supported by the relatively even
horizontal spread of data points across various image sizes.

In aiming to accurately identify human faces across a diverse array of group photos, the presence of a low
correlation coefficient signifies that image size does not substantially impact the linear number of objects
detected. This outcome suggests the potential for the face detection system to achieve consistent
performance irrespective of image size, advantageous for scalability. It further implies that tuning the
system’s algorithm specifically for image size may not be necessary to enhance accuracy, precision, and
recall. However, this also highlights the necessity for further exploration of other factors that might affect
detetction performance, such as image resolution, face occlusions, and the varying densities of people
within images. Such analysis is critical in the refinement process of the system, aiming to ensure robust
face detection under diverse conditions and contribute towards the system's real-time performance and
computational efficiency.


The figure below is a violin plot, utilized to illustrate the distribution of the number of objects detected
across various image sizes.

This violin plot discloses the density of detected objects across a spectrum of image sizes. Each violin
corresponds to a distinct image size, with the violin’s width reflecting the density of data points at
different detected object counts. Additionally, the plot incorporates box plots within the violins,
showcasing the median, interquartile range, and any outliers in the number of objects detected for each
size category.

The figure below provides insights into the variability and distribution of face detections across differing
image sizes, which is essential for the assurance that the face detection system is capable of consistently
identifying faces, irrespective of the photo’s dimensions. For instance, broader distributions of detetcted
objects for certain sizes may suggest variability in the algorithm's performance for those dimensions, while
narrower distributions could indicate consistent performance. By pinpointing image sizes with increased
variability or potential outliers in detection numbers, the project team can earmark these dimensions for
heightened scrutiny and enhancement, thus bolstering the system's robustness and dependability. Such
insights are crucial for propelling the project's objectives forward, aiming for superior accuracy and
precision in real-time face identification, especially in photos characterized by varying conditions and
densities.


The figure below is a hexbin plot, commonly referred to as a heatmap, which effectively demonstrates
the concentration of identified items in relation to the size of the image.

The heatmap displays hexagonal bins that represent areas with different densities, indicating a link
between the number of identified items and the size of the image. The hexagons' color intensity
corresponds to the number of observations in each bin, with deeper colors indicating a larger
concentration of photos for a set count of identified objects, and lighter colors indicating lower densities.
The plot demonstrates a positive correlation between smaller image sizes and a higher concentration of
images with a lower number of detected objects.

The figure below illustrates the concentration and distribution of detections across various image sizes.
This information is crucial for the project's goal of improving face detection performance across a wide
range of image sizes. The heatmap provides a strategic roadmap for directing optimization efforts by
highlighting areas of higher or lower challenge for the face identification algorithm. For instance, the
findings obtained from the heatmap could guide the project in improving detection algorithms for image
sizes that currently have low numbers of detections. This would help ensure that the system performs
well for a wide range of group photo compositions. Optimization is crucial for meeting the project's goals,
particularly in maintaining computing economy while attaining high levels of accuracy and precision in
real-time applications across different environments.


## 3.3 Discussion

Our current result still requires fine tuning, however, by the semester's end, we aim to further optimize
our system for real-time processing and explore the incorporation of additional contextual cues (e.g., body
posture, group dynamics) to enhance detection accuracy and applicability in more nuanced scenarios.

## 4. Related Work

Prior work by Shao, Shuai, et al[2] addressed the challenge of accurately detecting humans in crowded
scenes, where individuals are often partially occluded, making detection difficult. Their method involved
creating and utilizing the CrowdHuman[1] dataset, which is specifically designed with a focus on crowded
scenarios. The dataset provides detailed annotations for human heads, visible bodies, and full bodies,
enabling the development and evaluation of detection models that can better handle the complexities of
crowded environments. Through extensive experiments, the authors demonstrate the effectiveness of
their approach in improving detection performance in densely populated images.

The paper "Double Anchor R-CNN for Human Detection in a Crowd" [3] addresses the challenge of
detecting humans in crowded scenes, where occlusion significantly hampers detection accuracy. Their
proposed method, Double Anchor R-CNN, leverages a novel architecture that detects human heads and
bodies simultaneously by using double anchors for each person, aiming to mitigate occlusion issues. They
introduce a proposal crossover strategy and feature aggregation to enhance proposal quality and
detection reliability, respectively. Joint Non-Maximum Suppression (NMS) is developed for effective post-
processing. Their method achieves state-of-the-art results on several crowded human detection datasets,
demonstrating its effectiveness in improving human detection performance in crowded scenarios.


Building on the work of Shao et al.[2] and the Double Anchor R-CNN[3], we propose to enhance model
performance using convolutional neural networks (CNNs) with a focus on improving accuracy in detecting
humans in crowded scenes. Our method aims to integrate advanced CNN architectures with attention
mechanisms to better identify and differentiate individuals, even in densely populated areas.
Furthermore, we are considering the application of clustering techniques to group detected individuals,
potentially improving the handling of occlusions and interactions among people. This approach seeks to
refine detection capabilities and address the complexities of crowded environments more effectively.

## 5. Next Steps

- We will continue with the below points:
    o Further data cleaning is a top priority as some issues, like inconsistent data was not
       visible to us earlier.
    o Improve the results and accuracy of the body detection model.
    o Improve and fix model errors of the MTCNN face detection model.
    o We will perform some visualizations to compare our results based on different model
       parameters.
    o We have already incorporated a GPU in our project, we will target later on to run our
       model training loop on an AMD GPU.
- Our project consists of two major parts, face and body detection which allowed us to split the
    tasks to sub task per pair:
       o Face detection: Omar and Sepideh
       o Body detection: Rafael and Onyinye

## 5. Code and Dataset

We will share our code via GitHub as we have used it as our main collaboration platform for the project,
as for the dataset it is quite large to be shared online, so please find the following link.

## 6. Conclusion

We have gotten over our first steps in our project with data ingestion and visualization, this provided us
with insights on the data and how to handle it properly. We have researched various models and tested
benchmark models against our dataset, with the knowledge at hand we will move forward with
enhancing the models by retraining them.

## References

### [1] CrowdHuman Dataset

### [2] Shao, S., Zhao, Z., Li, B., Xiao, T., Yu, G., Zhang, X., & Sun, J. (2018). Crowdhuman: A

### benchmark for detecting human in a crowd. arXiv 2018. arXiv preprint arXiv:1805..


### [3] Zhang, K., Xiong, F., Sun, P., Hu, L., Li, B., & Yu, G. (2019). Double Anchor R-CNN for

### Human Detection in a Crowd. ArXiv. /abs/1909.

### [4] [1604.02878] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

(arxiv.org)

### [5] facenet-pytorch/README_cn.md at master · timesler/facenet-pytorch (github.com)


