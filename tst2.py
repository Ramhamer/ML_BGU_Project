import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.utils import class_weight
import numpy as np

# Define custom dataset
class LungCTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.with_covid = datasets.ImageFolder(root=data_dir + "/COVID-19", transform=transform)
        self.without_covid = datasets.ImageFolder(root=data_dir + "/Non-COVID-19", transform=transform)
        self.data = self.with_covid.samples + self.without_covid.samples
        self.targets = [1]*len(self.with_covid) + [0]*len(self.without_covid)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx < len(self.with_covid):
            return self.with_covid[idx][0], 1  # "with_covid"
        else:
            return self.without_covid[idx - len(self.with_covid)][0], 0  # "without_covid"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = LungCTDataset(data_dir='/home/oury/Documents/Ram/AI Project/GIT/COVID-19_Lung_CT_Scans', transform=transform)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

# Calculate class weights for imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(dataset.targets),
    dataset.targets
)
class_weights = torch.FloatTensor(class_weights)

# Define model
class LungCTClassifier(nn.Module):
    def __init__(self):
        super(LungCTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 128 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate the model, loss function, and optimizer
model = LungCTClassifier()
criterion = nn.BCELoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        
    # Validate the model
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs.squeeze(), labels.float()).item()
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {(correct/total)*100}%")
