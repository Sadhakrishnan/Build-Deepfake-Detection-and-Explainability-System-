import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        # Using ResNet50 as a robust substitute for XceptionNet (which isn't natively in torchvision)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer for binary classification (Real vs Fake)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

    def predict(self, image_tensor):
        """
        Predicts if a given normalized image tensor is Fake.
        Returns: probability of being fake [0.0, 1.0]
        """
        self.eval()
        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0) # Add batch dimension
                
            output = self.forward(image_tensor)
            return output.item()
