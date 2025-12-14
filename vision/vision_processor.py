"""
Vision processing module for drone CV-based tank detection
Processes camera images to extract features for RL agent
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Tuple, Optional


class VisionEncoder(nn.Module):
    """
    CNN encoder for processing camera images.
    Extracts visual features from RGB images for tank detection.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 output_dim: int = 256,
                 image_size: Tuple[int, int] = (480, 640)):
        """
        Initialize vision encoder.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            output_dim: Dimension of output feature vector
            image_size: (height, width) of input images
        """
        super().__init__()
        
        self.image_size = image_size
        self.input_channels = input_channels
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate flattened size after conv layers
        # This is approximate - adjust based on actual feature map size
        conv_output_size = self._calculate_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the size of flattened conv output."""
        # Approximate calculation based on image size and conv layers
        # In practice, you'd do a forward pass with dummy input
        h, w = self.image_size
        # After conv layers: approximate size
        h_out = h // 8  # After stride 4 and stride 2
        w_out = w // 8
        return 256 * h_out * w_out
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            Feature vector [batch_size, output_dim]
        """
        # Normalize images to [0, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0
        
        # Extract features
        conv_features = self.conv_layers(images)
        features = self.fc_layers(conv_features)
        
        return features


class TankDetector(nn.Module):
    """
    Tank detection module using object detection or segmentation.
    Can be used to provide additional information to the RL agent.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 2):  # Background + Tank
        """
        Initialize tank detector.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of classes (background + tank)
        """
        super().__init__()
        
        # Simple segmentation head (can be replaced with more sophisticated models)
        self.backbone = VisionEncoder(input_channels, output_dim=256)
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect tank in images.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            segmentation_mask: Segmentation mask [batch_size, num_classes, h, w]
            features: Feature vector [batch_size, feature_dim]
        """
        features = self.backbone(images)
        # Note: This is simplified - in practice, you'd need to reshape features
        # for segmentation head
        return features, features


class VisionProcessor:
    """
    High-level vision processing pipeline.
    Handles image preprocessing and feature extraction.
    """
    
    def __init__(self, 
                 encoder: Optional[VisionEncoder] = None,
                 image_size: Tuple[int, int] = (480, 640),
                 device: str = "cuda:0"):
        """
        Initialize vision processor.
        
        Args:
            encoder: Optional pre-trained vision encoder
            image_size: Expected image size (height, width)
            device: Device to run processing on
        """
        self.device = device
        self.image_size = image_size
        
        if encoder is None:
            self.encoder = VisionEncoder(image_size=image_size).to(device)
        else:
            self.encoder = encoder.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def process_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process camera images to extract features.
        
        Args:
            images: Raw camera images [batch_size, height, width, channels] or 
                   [batch_size, channels, height, width]
            
        Returns:
            Feature vectors [batch_size, feature_dim]
        """
        # Ensure images are on correct device
        images = images.to(self.device)
        
        # Convert to [batch, channels, height, width] if needed
        if images.dim() == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        # Resize if needed
        if images.shape[2:] != self.image_size:
            images = torch.nn.functional.interpolate(
                images, size=self.image_size, mode='bilinear', align_corners=False
            )
        
        # Extract features
        with torch.no_grad():
            features = self.encoder(images)
        
        return features
    
    def get_tank_position_from_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Estimate tank position from images (optional helper method).
        This could use object detection or other CV techniques.
        
        Args:
            images: Camera images
            
        Returns:
            Estimated tank positions [batch_size, 3] (relative to drone)
        """
        # Placeholder - in practice, use object detection or depth estimation
        # to determine tank position in camera frame
        batch_size = images.shape[0]
        return torch.zeros((batch_size, 3), device=self.device)

