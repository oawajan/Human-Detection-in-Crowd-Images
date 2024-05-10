from Libraries import *


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

