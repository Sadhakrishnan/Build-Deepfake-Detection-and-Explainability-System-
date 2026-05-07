import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # The backward hook returns a tuple, usually gradient w.r.t output is what we want
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap for a given input tensor.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        if output.shape[1] == 1:
            # Binary classification (sigmoid output or single logit)
            score = output[0, 0]
        else:
            # Multi-class
            score = output[0, class_idx]
            
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU to keep only positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam

def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays heatmap on an image.
    Image should be an RGB numpy array [0-255].
    Heatmap should be a 2D numpy array [0-1].
    """
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(image, alpha, heatmap_colored, 1 - alpha, 0)
    return superimposed_img
