"""
Convolutional Autoencoder (CAE) Model for Weld Defect Detection

Architecture:
- Encoder: 3 conv blocks with max pooling
- Latent space: Compressed representation
- Decoder: 3 conv transpose blocks for reconstruction

Trained on normal (ND) images only.
Anomalies (defects) produce high reconstruction error.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """Deconvolutional block: ConvTranspose -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class CAE(nn.Module):
    """
    Convolutional Autoencoder for Anomaly Detection
    
    Input: 224x224x3 RGB image
    Latent: 28x28x128 feature map (or flattened to 128-dim vector)
    Output: 224x224x3 reconstructed image
    """
    
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)    # 224 -> 224
        self.pool1 = nn.MaxPool2d(2, 2)           # 224 -> 112
        
        self.enc2 = ConvBlock(32, 64)             # 112 -> 112
        self.pool2 = nn.MaxPool2d(2, 2)           # 112 -> 56
        
        self.enc3 = ConvBlock(64, 128)            # 56 -> 56
        self.pool3 = nn.MaxPool2d(2, 2)           # 56 -> 28
        
        # Bottleneck
        self.bottleneck = ConvBlock(128, latent_dim)  # 28 -> 28
        
        # Decoder
        self.dec3 = DeconvBlock(latent_dim, 128)  # 28 -> 56
        self.dec2 = DeconvBlock(128, 64)          # 56 -> 112
        self.dec1 = DeconvBlock(64, 32)           # 112 -> 224
        
        # Output layer (no Sigmoid - working with ImageNet normalized values)
        self.output = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
    
    def encode(self, x):
        """Encode input to latent representation"""
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.bottleneck(x)
        return x
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        x = self.dec3(z)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.output(x)
        return x
    
    def forward(self, x):
        """Forward pass: encode then decode"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction
    
    def get_reconstruction_error(self, x, reduction='mean'):
        """
        Calculate reconstruction error for anomaly detection
        
        Args:
            x: Input image tensor
            reduction: 'mean' for single value, 'none' for pixel-wise
            
        Returns:
            error: Reconstruction error (MSE)
            reconstruction: Reconstructed image
        """
        with torch.no_grad():
            reconstruction = self.forward(x)
            
            if reduction == 'mean':
                # Mean squared error across all pixels
                error = torch.mean((x - reconstruction) ** 2, dim=[1, 2, 3])
            elif reduction == 'none':
                # Pixel-wise error (for heatmap visualization)
                error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
                
        return error, reconstruction


class CAELarge(nn.Module):
    """
    Larger CAE with more capacity for complex patterns
    
    Input: 224x224x3 RGB image
    Output: 224x224x3 reconstructed image
    """
    
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: 56 -> 28
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4: 28 -> 14
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 5: 14 -> 7
            nn.Conv2d(512, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Block 1: 7 -> 14
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 2: 14 -> 28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 3: 28 -> 56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 56 -> 112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 5: 112 -> 224 (no Sigmoid - working with ImageNet normalized values)
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction
    
    def get_reconstruction_error(self, x, reduction='mean'):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstruction = self.forward(x)
            
            if reduction == 'mean':
                error = torch.mean((x - reconstruction) ** 2, dim=[1, 2, 3])
            elif reduction == 'none':
                error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
                
        return error, reconstruction


class CAESmall(nn.Module):
    """
    Smaller/Simpler CAE to avoid overfitting on small datasets.
    
    Key differences from CAE:
    - Fewer filters (16-32-64 vs 32-64-128)
    - Smaller bottleneck (64 vs 128)
    - ~110K parameters vs ~886K parameters
    
    Use this when the standard CAE reconstructs defects too well.
    """
    
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder - fewer filters
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 112 -> 56
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 56 -> 28
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Bottleneck: 28 -> 14
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 14 -> 28
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 28 -> 56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 56 -> 112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 112 -> 224
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def get_reconstruction_error(self, x, reduction='mean'):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstruction = self.forward(x)
            
            if reduction == 'mean':
                error = torch.mean((x - reconstruction) ** 2, dim=[1, 2, 3])
            elif reduction == 'none':
                error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
                
        return error, reconstruction


class CAETiny(nn.Module):
    """
    Even simpler CAE for very small datasets or extreme overfitting cases.
    
    Key features:
    - Minimal filters (8-16-32)
    - Very aggressive bottleneck (7x7x32)
    - ~30K parameters
    - Forces model to learn only the most essential features
    """
    
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder - minimal filters
        self.encoder = nn.Sequential(
            # 224 -> 56 (stride 4)
            nn.Conv2d(in_channels, 8, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 56 -> 14
            nn.Conv2d(8, 16, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 14 -> 7
            nn.Conv2d(16, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(latent_dim, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 14 -> 56
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 56 -> 224
            nn.ConvTranspose2d(8, in_channels, kernel_size=7, stride=4, padding=3, output_padding=3),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def get_reconstruction_error(self, x, reduction='mean'):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstruction = self.forward(x)
            
            if reduction == 'mean':
                error = torch.mean((x - reconstruction) ** 2, dim=[1, 2, 3])
            elif reduction == 'none':
                error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
                
        return error, reconstruction


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Test CAE
    model = CAE().to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    
    reconstruction = model(x)
    print(f"\nCAE Model:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {reconstruction.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test reconstruction error
    error, recon = model.get_reconstruction_error(x)
    print(f"  Reconstruction error: {error.mean().item():.6f}")
    
    # Test CAELarge
    model_large = CAELarge().to(device)
    reconstruction = model_large(x)
    print(f"\nCAELarge Model:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {reconstruction.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_large.parameters()):,}")
    
    # Test CAESmall
    model_small = CAESmall().to(device)
    reconstruction = model_small(x)
    print(f"\nCAESmall Model:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {reconstruction.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_small.parameters()):,}")
    
    # Test CAETiny
    model_tiny = CAETiny().to(device)
    reconstruction = model_tiny(x)
    print(f"\nCAETiny Model:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {reconstruction.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_tiny.parameters()):,}")
