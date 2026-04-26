import torch
import torch.nn as nn
import torch.nn.functional as F

class PCBDefectCNN(nn.Module):
    """
    A simple but robust CNN for binary classification of PCB defects.
    Input size: (3, 224, 224)
    """
    def __init__(self, num_classes=2):
        super(PCBDefectCNN, self).__init__()
        
        # Block 1: Low-level features
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 2: Middle-level features
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Block 3: High-level features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected layers
        # After 3 pooling layers (224 -> 112 -> 56 -> 28)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 64 * 28 * 28)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Test with a dummy tensor
    model = PCBDefectCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print("Model initialized successfully!")
