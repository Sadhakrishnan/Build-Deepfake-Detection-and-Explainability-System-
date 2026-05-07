import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .detector import DeepfakeDetector

def train_model(train_loader, val_loader, num_epochs=10, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector(pretrained=True).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")
        
    # Save checkpoint
    torch.save(model.state_dict(), 'deepfake_detector.pth')
    return model
