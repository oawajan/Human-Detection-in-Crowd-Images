from Libraries import *
from Evaluation import *
from MTCNN import *
from ResNet import *
from DataClass import *
from Visualizations import *


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    initpath = "./CrowdHuman_Dataset"  # Define the path to the dataset
    data = readdata(initpath, "annotation_train.odgt")
    images = readimages(data, initpath)
    displayimages(images)
    #noisy_images = augment_images_with_noise(images)
    #displayimages(noisy_images)  # Display augmented images
    #colortransformations(images)
    #pixelhistogram(images)
    #image_size_vs_objects(images, data)
    #testdetection(data, images)
    # Load the detection model
    # detection_model = load_detection_model()
    # Perform full body detection
    # body_detections_nms = detect_full_body_nms(images, detection_model)
    # ground_truth_boxes = [entry['gtboxes'] for entry in data[:len(images)]]
    # Evaluate the detections
    # evaluation_results = calculate_metrics(body_detections_nms, ground_truth_boxes)
    # print(evaluation_results)
    # Visualize detections
    # visualize_detections(images, body_detections_nms)
    # trainMTCNN(data, images)
