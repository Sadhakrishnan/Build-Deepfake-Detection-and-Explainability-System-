import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

class FaceDetector:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.mtcnn = MTCNN(
            image_size=224, margin=20, keep_all=False,
            post_process=False, device=self.device
        )
        
    def detect_and_crop(self, image):
        """
        Detects a face in an image and returns the cropped face tensor.
        Input: image (numpy array or PIL Image)
        Output: torch tensor of cropped face, or None if no face found
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # MTCNN returns a tensor if post_process=False
        face_tensor = self.mtcnn(image)
        
        if face_tensor is not None:
            # Normalize to 0-1 range for typical CNNs if not post_processed
            face_tensor = face_tensor / 255.0
            
            # Typical normalization for ImageNet
            normalize = torch.nn.Sequential(
                torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
            # Add batch dimension to apply normalization, then remove it
            face_tensor = normalize(face_tensor.unsqueeze(0)).squeeze(0)
            
            return face_tensor
        return None

    def detect_multiple(self, image):
        self.mtcnn.keep_all = True
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        faces = self.mtcnn(image)
        self.mtcnn.keep_all = False
        return faces
