from Libraries import *

class ImageDataset(Dataset):
    def __init__(self, target_transform=None):
        self.init_path = f".\\CrowdHuman_Dataset\\"
        self.image_labels = self.read_annotations()

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image = self.load_image(self.image_labels[index])
        return image

    def read_annotations(self) -> list:
        datafile = []
        annotations = []
        with open(f"{self.init_path}annotation_train.odgt") as file:
            for line in file:
                datafile.append(json.loads(line))
        return datafile

    def load_image(self, image_annotation):
        ID = image_annotation['ID']
        paths = (f"{self.init_path}CrowdHuman_train01\\Images\\{ID}.JPG",
                 f"{self.init_path}CrowdHuman_train02\\Images\\{ID}.JPG",
                 f"{self.init_path}CrowdHuman_train03\\Images\\{ID}.JPG")
        try:
            for path in paths:
                img = read_image(path)
                if img is not None:
                    img = img.permute(1, 2, 0)
                    plt.imshow(img)
                    plt.title(ID)
                    plt.show()
                    print(type(img))
                    return img
        except Exception as e:
            print(f"Error reading image from: {str(e)}")


def trainMTCNN(data, images) -> list:

    if (torch.cuda.is_available()):
        print(f"GPU is available {torch.cuda.is_available()}, Version {torch.version.cuda}")
        device = torch.device('cuda')
    else:
        print(f"Using CPU")
        device = torch.device('cpu')
    # initialize the MTCNN model and set it to use the GPU
    model = torchmtcnn(device=device)
    # print(model)
    print("##################freeze pretrained model layers except  one#############################")
    # freeze pretrained model layers
    for name, param in torchmtcnn.named_parameters(model):
        # print(f"name: {name}")
        # print(f"param: {param}")
        if 'pnet.conv1' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("##################              load dataset                #############################")
    dataset = ImageDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # epoch count
    epochs = 10
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

def collate_fn(batch):
    filtered_batch = []
    for item in batch:
        if item is not None:
            print(type(item))
            filtered_batch.append(item)
    return default_collate(filtered_batch)