import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MTCNN import *
from Libraries import *

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)

        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 54 * 54, 64)
        self.fc2 = nn.Linear(64, 5)
        self.fc_bbox = nn.Linear(64 * 54 * 54, 4)  # Adjust num_bbox_output_units accordingly

    def forward(self, images):
        # Processing image data
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 54 * 54)

        # Processing non-uniform labels
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))

        # Final output layers for bounding box coordinates and labels
        bbox_output = self.fc_bbox(x)  # Output bounding box coordinates
        labels_output = self.fc_labels(x)  # Output labels

        return bbox_output, labels_output


# Instantiate the model
model = CustomCNN()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a random image tensor and non-uniform labels tensor for testing
dataset = ImageDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Define loss function
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.float()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{10}], Loss: {epoch_loss:.4f}')

print('Training finished!')