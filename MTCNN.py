import torch

from Libraries import *
from Evaluation import IoU


def collate_fn(batch):
    filtered_batch = [item for item in batch if item is not None]
    images, labels = zip(*filtered_batch)
    max_num_boxes = max(len(l) for l in labels)
    padded_labels = []
    for l in labels:
        current_size = l.size(0)
        if current_size < max_num_boxes:
            pad_size = max_num_boxes - current_size
            pad_tensor = torch.zeros(pad_size, *l.shape[1:])
            padded_labels.append(torch.cat([l, pad_tensor], dim=0))
        elif current_size > max_num_boxes:
            padded_labels.append(l[:max_num_boxes])
        else:
            padded_labels.append(l)
    images = torch.stack(images, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    print(len(images), len(padded_labels))
    return images, padded_labels


def post_process_detections(boxes, scores, score_threshold=0.5, iou_threshold=0.3):
    if isinstance(scores, torch.Tensor):
        scores = scores.tolist()
    elif isinstance(scores, list):
        scores = [score for sublist in scores for score in sublist] if scores and isinstance(scores[0], list) else scores
    high_confidence_idxs = [i for i, score in enumerate(scores) if score > score_threshold]
    filtered_boxes = [boxes[i] for i in high_confidence_idxs]
    filtered_scores = [scores[i] for i in high_confidence_idxs]
    final_idxs = nms(filtered_boxes, filtered_scores, iou_threshold=iou_threshold)
    final_boxes = [filtered_boxes[i] for i in final_idxs]

    return final_boxes


def nms(boxes, scores, iou_threshold=0.5):
    scores = torch.tensor(scores) if isinstance(scores, list) else scores
    idxs = torch.argsort(scores, descending=True)
    keep = []
    while idxs.numel() > 0:
        current = idxs[0]
        keep.append(current.item())
        if idxs.numel() == 1:
            break
        current_box = boxes[current]
        remaining_boxes = boxes[idxs[1:]]
        ious = iou(current_box, remaining_boxes)

        idxs = idxs[1:][ious < iou_threshold]

    return keep


def iou(box1, boxes):
    inter_x1 = torch.max(box1[0], boxes[:, 0])
    inter_y1 = torch.max(box1[1], boxes[:, 1])
    inter_x2 = torch.min(box1[2], boxes[:, 2])
    inter_y2 = torch.min(box1[3], boxes[:, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    union_area = box1_area + boxes_area - inter_area
    return inter_area / union_area


def process_rnet_outputs(image, boxes, scores, score_threshold=0.5, resize_size=48):
    idxs = scores > score_threshold
    filtered_boxes = boxes[idxs]
    filtered_scores = scores[idxs]
    final_boxes = filtered_boxes
    crops = []
    for box in final_boxes:
        cropped_img = image.crop((box[0], box[1], box[2], box[3]))
        resized_img = cropped_img.resize((resize_size, resize_size), Image.BILINEAR)
        tensor_img = F.to_tensor(resized_img)
        crops.append(tensor_img)
    if crops:
        batch = torch.stack(crops)
    else:
        batch = torch.empty(0, 3, resize_size, resize_size)  # handle the case of no boxes

    return batch


class ImageDataset(Dataset):
    def __init__(self, target_transform=None):
        self.init_path = f".\\CrowdHuman_Dataset\\"
        self.image_data = self.read_annotations()
        self.target_transform = target_transform
        self.mtcnn = torchmtcnn

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image, label = self.load_image_and_labels(index)
        return image, label

    def read_annotations(self) -> list:
        data = []
        with open(os.path.join(self.init_path, "annotation_train.odgt")) as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def load_image_and_labels(self, index):
        instance = self.image_data[index]
        ID = instance["ID"]
        img = None
        for folder in ["Images", "Images 2", "Images 3"]:
            img_path = os.path.join(self.init_path, folder, f"{ID}.JPG")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((160, 160))
                img = transforms.ToTensor()(img)
                break
        if img is None:
            return None, None

        # Extract labels
        boxes = []
        for people in instance["gtboxes"]:
            box = people['hbox']
            box.append(1)
            boxes.append(box)
        while len(boxes) < 200:
            boxes.append([-1, -1, -1, -1, 0])
        label = torch.tensor(boxes)

        return img, label


def trainMTCNN():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchmtcnn(device=device)
    p_model, r_model, o_model = model.pnet, model.rnet, model.onet
    for net in [p_model, r_model, o_model]:
        for param in net.parameters():
            param.requires_grad = True
    dataset = ImageDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_function = torch.nn.BCEWithLogitsLoss()

    epochs = 10
    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                p_outputs = p_model(images)
                p_tensor = F.avg_pool2d(p_outputs[0][:, 3, :, :], kernel_size=3, stride=3)
                r_input = F.interpolate(p_tensor.unsqueeze(1).expand(-1, 3, -1, -1), size=(24, 24), mode='bilinear', align_corners=False)
                batch_boxes, batch_scores = r_model(r_input)

            final_detections = []
            for image_idx, (boxes, scores) in enumerate(zip(batch_boxes, batch_scores)):
                processed_boxes = post_process_detections(boxes, scores)
                for box in processed_boxes:
                    final_detections.append((image_idx, box))

            o_inputs = []
            for image_idx, bbox in final_detections:
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.tolist()
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cropped_image = images[image_idx][:, max(0, y1):min(images.shape[2], y2), max(0, x1):min(images.shape[3], x2)]
                    resized_image = F.interpolate(cropped_image.unsqueeze(0), size=(48, 48), mode='bilinear', align_corners=False)
                    o_inputs.append(resized_image.squeeze(0))

            if o_inputs:
                o_inputs = torch.stack(o_inputs).to(device)
                o_outputs = o_model(o_inputs)
                loss = loss_function(o_outputs, labels)
            else:
                print("Warning: No detections to process.")
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return o_model
