from Libraries import *

class CrowdHuman(Dataset):
    def __init__(self, root=f"./CrowdHuman_Dataset\\Images\\", ann_file="./CrowdHuman_Dataset\\annotation_train.odgt",
                 remove_images_without_annotations=True, *, order=None):
        super().__init__()
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root
        self.order = order
        self.supported_order = (
            "image",
            "boxes",
            "vboxes",
            "hboxes",
            "boxes_category",
            "info",
        )

        print('load annotation file: ', ann_file)
        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            self.imgs[img["id"]] = img

        self.imgs_with_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            self.imgs_with_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        self.ids = list(sorted(self.imgs.keys()))  # A list contains keys

        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.imgs_with_anns[img_id]
                if len(anno) == 0:
                    del self.imgs[img_id]
                    del self.imgs_with_anns[img_id]
                else:
                    ids.append(img_id)

            self.ids = ids
        print("load with order", self.order)

    def __getitem__(self, index):
        img_id = self.ids[index]
        anno = self.imgs_with_anns[img_id]

        target = []
        for k in self.order:
            if k == "image":
                file_name = self.imgs[img_id]["file_name"]
                path = os.path.join(self.root, file_name)
                image = cv2.imread(path, cv2.IMREAD_COLOR)  # BRG
                target.append(image)
            elif k == "boxes":
                boxes = [obj["bbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                # transfer boxes from xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "vboxes":
                boxes = [obj["vbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "hboxes":
                boxes = [obj["hbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "boxes_category":
                boxes_category = [obj["category_id"] for obj in anno]
                boxes_category = np.array(boxes_category, dtype=np.int32)
                target.append(boxes_category)
            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"], info["file_name"]]
                target.append(info)

        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info


def trainMTCNN2() -> list:

    if (torch.cuda.is_available()):
        print(f"GPU is available {torch.cuda.is_available()}, Version {torch.version.cuda}")
        device = torch.device('cuda')
    else:
        print(f"Using CPU")
        device = torch.device('cpu')
    # initialize the MTCNN model and set it to use the GPU
    model = torchmtcnn(device=device)
    # sum = summary(model, (3, 224, 224))
    # print(sum)
    # print(help(torchmtcnn))
    print(model)
    print("##################freeze pretrained model layers except  output layer#############################")
    # freeze pretrained model layers
    # for name, param in torchmtcnn.named_parameters(model):
    #     print(f"name: {name}")
    #     print(f"param: {param}")
    #     if 'pnet.conv1' not in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    # for name, param in model.named_parameters():
    #     if 'box' in name or 'cls' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    for name, param in model.named_parameters():
        if 'onet' in name:  # Identify parameters belonging to the ONet
            param.requires_grad = True  # Unfreeze the parameter
        else:
            param.requires_grad = False  # Freeze the parameter for all other layers

    print("##################              load dataset                #############################")
    dataset = CrowdHuman()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # loss function
    loss_function = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, targets)
            loss = loss_function(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
