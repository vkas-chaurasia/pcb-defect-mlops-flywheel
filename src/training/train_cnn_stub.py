import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import mlflow
import yaml
import os
from PIL import Image
from model import PCBDefectCNN

# 1. Load Configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Dataset Class
class PCBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        # Simulated label for demo (usually you'd have a CSV or subfolders)
        # 0: No Defect, 1: Defect
        label = 1 if "defect" in img_name.lower() else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train():
    # 3. Setup MLflow
    mlflow.set_experiment("PCB_Defect_Detection")
    
    with mlflow.start_run():
        # Log Parameters
        mlflow.log_params(config['training'])
        
        # 4. Prepare Data
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = PCBDataset(config['training']['data_dir'], transform=transform)
        
        if len(dataset) == 0:
            print("No data found in processed directory. Skipping training.")
            return
            
        train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
        
        # 5. Initialize Model
        model = PCBDefectCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
        # 6. Training Loop
        model.train()
        for epoch in range(config['training']['epochs']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {avg_loss:.4f}")
            
        # 7. Save Model
        os.makedirs(config['training']['model_dir'], exist_ok=True)
        model_path = os.path.join(config['training']['model_dir'], "model.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Training Complete. Model saved to {model_path}")

if __name__ == "__main__":
    train()
