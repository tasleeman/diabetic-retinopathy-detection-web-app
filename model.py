import torch.nn as nn
import torch.nn.functional as F

class DRModel(nn.Module):
    def __init__(self, num_classes=5):
        super(DRModel, self).__init__()
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Input shape: [batch, 3, 224, 224]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch, 128, 28, 28]
        x = x.view(-1, 128 * 28 * 28)         # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x