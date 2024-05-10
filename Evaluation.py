from Libraries import *

'''
Image Size vs. Objects Detected
'''


def image_size_vs_objects(images, data) -> None:
    sizes = [img.shape[0] * img.shape[1] for img in images]
    objects = [len(entry['gtboxes']) for entry in data[:len(images)]]

    # Zip the two lists together, sort them by image size, and unzip them back
    sorted_data = sorted(zip(sizes, objects), key=lambda x: x[0])
    sorted_sizes, sorted_objects = zip(*sorted_data)

    # Scatter Plot with Ordered Data
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_sizes, sorted_objects, color='blue')
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


def IoU(data,predictions)->list:
    results = []
    print(data)
    for i, prediction in enumerate(predictions):
        print(prediction)
        temp = []
        for boxes in prediction:
            max_IoU = 0
            for box in range(len(data[i])):
                x_intersection = max(boxes[0],
                                     data[i][0])

                y_intersection = max(boxes[1],
                                     data[i][1])

                w_intersection = min(boxes[0] + boxes[2],
                                     data[i][0] + data[i][2]) - x_intersection

                h_intersection = min(boxes[1] + boxes[3],
                                     data[i][1] + data[i][3]) - y_intersection
                area_intersection = max(0, w_intersection) * max(0, h_intersection)

                area_union = (boxes[2] * boxes[3]
                              + data[i][2] * data[i][3]
                              - area_intersection)

                if area_union == 0:
                    IoU = 0
                else:
                    IoU = area_intersection / area_union

                max_IoU = max(max_IoU, IoU)
            results.append(temp)
    print(results)
    return results


def IoU_metric(data, predictions) -> list:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    results = []

    for i, prediction in enumerate(predictions):
        temp = []
        if len(data[i]['gtboxes']) == 0:
            false_positives += len(prediction)
            false_negatives += 0
        elif len(prediction) == 0:
            false_positives += 0
            false_negatives += len(data[i]['gtboxes'])
        else:
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

                    if area_union == 0:
                        IoU = 0
                    else:
                        IoU = area_intersection / area_union

                    max_IoU = max(max_IoU, IoU)
                if max_IoU < 0.01:
                    false_positives += 1
                else:
                    true_positives += 1

            false_negatives += max(0, len(data[i]['gtboxes']) - true_positives)

        results.append(temp)

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



def IoU2(data, predictions) -> list:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    i = 0
    results = []
    for prediction in predictions:
        temp = []
        if len(data[i]['gtboxes']) == 0:
            false_positives += len(predictions)

        if len(predictions) == 0:
            false_negatives += len(data[i]['gtboxes'])

        if len(prediction) > len(data[i]['gtboxes']):
            false_negatives += len(prediction) - len(data[i]['gtboxes'])

        if len(prediction) < len(data[i]['gtboxes']):
            false_positives += len(data[i]['gtboxes']) - len(prediction)

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
                if max_IoU < 0.2:
                    true_positives += 1
                else:
                    false_negatives += 1

            temp.append(max_IoU)
            i += 1
        results.append(temp)
        print (results)
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
#    return results


def testdetection(data,images) -> None:
    results = []
    IoUresults = []

    results.append(testopencv(images))
    results.append(testMTCNN(images))
    #torch_results = testtorch(images)

    for result in results:
        IoUresults.append(IoU(data, result))


    return IoUresults


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
