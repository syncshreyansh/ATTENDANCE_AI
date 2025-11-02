#!/usr/bin/env python3
"""
Train anti-spoofing CNN on CASIA-SURF, Replay-Attack, or custom dataset
Dataset structure:
  train/
    live/*.jpg
    photo/*.jpg
    screen/*.jpg
  val/
    live/*.jpg
    photo/*.jpg
    screen/*.jpg
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import argparse

class SpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 0: live, 1: photo, 2: screen
        for cls_idx, cls_name in enumerate(['live', 'photo', 'screen']):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(cls_dir, img_name), cls_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(data_dir, epochs, output_path):
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = SpoofDataset(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = SpoofDataset(os.path.join(data_dir, 'val'), val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc
            }, output_path)
            print(f"  → Saved best model (acc={val_acc:.2f}%)")
    
    print(f"Training complete. Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output', required=True, help='Output model path')
    args = parser.parse_args()
    
    train_model(args.dataset, args.epochs, args.output)