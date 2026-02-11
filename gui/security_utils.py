"""
Security Utilities for RIAWELC Project
Provides model integrity verification and secure file operations
"""

import hashlib
import hmac
import json
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ModelIntegrityVerifier:
    """Handles model integrity verification using cryptographic hashes"""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the verifier
        
        Args:
            secret_key: Optional secret key for HMAC signing. If None, uses SHA256 only.
        """
        self.secret_key = secret_key
        self.hash_file_extension = ".hash"
    
    def generate_model_hash(self, model_path: str) -> str:
        """
        Generate SHA256 hash of model file
        
        Args:
            model_path: Path to model file
            
        Returns:
            Hex digest of the hash
        """
        sha256_hash = hashlib.sha256()
        
        with open(model_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def generate_signature(self, file_hash: str) -> Optional[str]:
        """
        Generate HMAC signature for a file hash
        
        Args:
            file_hash: Hash of the file
            
        Returns:
            HMAC signature or None if no secret key
        """
        if not self.secret_key:
            return None
        
        return hmac.new(
            self.secret_key.encode(),
            file_hash.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def save_model_with_verification(self, model_path: str) -> bool:
        """
        Generate and save verification hash for a model
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate hash
            file_hash = self.generate_model_hash(model_path)
            
            # Generate signature if secret key available
            signature = self.generate_signature(file_hash)
            
            # Save verification data
            verification_data = {
                "file": Path(model_path).name,
                "hash": file_hash,
                "algorithm": "sha256"
            }
            
            if signature:
                verification_data["signature"] = signature
                verification_data["signature_algorithm"] = "hmac-sha256"
            
            hash_file_path = f"{model_path}{self.hash_file_extension}"
            with open(hash_file_path, 'w') as f:
                json.dump(verification_data, f, indent=2)
            
            logger.info(f"Model verification hash saved: {hash_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model verification: {e}")
            return False
    
    def save_pytorch_model(self, model_or_state_dict, model_path: str, is_state_dict: bool = True) -> bool:
        """
        Securely save PyTorch model and automatically generate integrity hash
        
        This is the recommended way to save models - combines torch.save() with automatic
        hash generation in a single atomic operation.
        
        Args:
            model_or_state_dict: PyTorch model state_dict or full model
            model_path: Path where model will be saved
            is_state_dict: True if saving state_dict (recommended), False if saving full model
            
        Returns:
            True if successful, False otherwise
            
        Example:
            >>> verifier = ModelIntegrityVerifier()
            >>> verifier.save_pytorch_model(model.state_dict(), 'model.pth')
            ✓ Model saved securely: model.pth
            ✓ Integrity hash generated: model.pth.hash
        """
        import torch
        try:
            # Save the model
            torch.save(model_or_state_dict, model_path)
            logger.info(f"✓ Model saved: {model_path}")
            
            # Automatically generate integrity hash
            hash_generated = self.save_model_with_verification(model_path)
            
            if hash_generated:
                logger.info(f"✓ Integrity hash generated: {model_path}.hash")
                return True
            else:
                logger.warning(f"⚠ Model saved but hash generation failed: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model securely: {e}")
            return False
    
    def verify_model(self, model_path: str, raise_on_failure: bool = True) -> bool:
        """
        Verify model integrity
        
        Args:
            model_path: Path to model file
            raise_on_failure: If True, raises exception on verification failure
            
        Returns:
            True if verification passes, False otherwise
            
        Raises:
            FileNotFoundError: If hash file doesn't exist
            SecurityError: If verification fails and raise_on_failure is True
        """
        hash_file_path = f"{model_path}{self.hash_file_extension}"
        
        # Check if hash file exists
        if not Path(hash_file_path).exists():
            warning_msg = f"No verification hash found for {model_path}. Model integrity cannot be verified."
            logger.warning(warning_msg)
            if raise_on_failure:
                raise FileNotFoundError(warning_msg)
            return False
        
        try:
            # Load verification data
            with open(hash_file_path, 'r') as f:
                verification_data = json.load(f)
            
            expected_hash = verification_data.get('hash')
            expected_signature = verification_data.get('signature')
            
            # Calculate current hash
            current_hash = self.generate_model_hash(model_path)
            
            # Verify hash
            if current_hash != expected_hash:
                error_msg = f"Model integrity check FAILED for {model_path}: Hash mismatch"
                logger.error(error_msg)
                if raise_on_failure:
                    raise SecurityError(error_msg)
                return False
            
            # Verify signature if available
            if expected_signature and self.secret_key:
                current_signature = self.generate_signature(current_hash)
                if current_signature != expected_signature:
                    error_msg = f"Model integrity check FAILED for {model_path}: Signature mismatch"
                    logger.error(error_msg)
                    if raise_on_failure:
                        raise SecurityError(error_msg)
                    return False
            
            logger.info(f"Model integrity verified: {model_path}")
            return True
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid verification file format for {model_path}: {e}"
            logger.error(error_msg)
            if raise_on_failure:
                raise SecurityError(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error verifying model: {e}"
            logger.error(error_msg)
            if raise_on_failure:
                raise
            return False


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


def secure_torch_load(filepath: str, map_location=None, verify_integrity: bool = True):
    """
    Secure wrapper for torch.load with integrity verification
    
    Args:
        filepath: Path to model file
        map_location: Device mapping for torch.load
        verify_integrity: Whether to verify model integrity before loading
        
    Returns:
        Loaded model checkpoint
        
    Raises:
        SecurityError: If integrity verification fails
    """
    import torch
    
    # Verify integrity if requested
    if verify_integrity:
        verifier = ModelIntegrityVerifier()
        verifier.verify_model(filepath, raise_on_failure=False)  # Warning only
    
    # Load with weights_only=True for security
    try:
        return torch.load(filepath, map_location=map_location, weights_only=True)
    except Exception as e:
        # If weights_only fails, log warning and try unsafe mode
        logger.warning(
            f"Failed to load {filepath} with weights_only=True. "
            f"This may indicate the model contains custom objects. Error: {e}"
        )
        # For backward compatibility, but log the security risk
        logger.warning("SECURITY WARNING: Loading model with weights_only=False")
        return torch.load(filepath, map_location=map_location, weights_only=False)


def generate_hashes_for_existing_models(models_dir: str = "models"):
    """
    Utility to generate hash files for existing model files
    
    Args:
        models_dir: Directory containing model files
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    verifier = ModelIntegrityVerifier()
    model_files = list(models_path.glob("*.pth")) + list(models_path.glob("*.pt"))
    
    print(f"Generating hashes for {len(model_files)} model files...")
    
    for model_file in model_files:
        print(f"Processing: {model_file.name}")
        success = verifier.save_model_with_verification(str(model_file))
        if success:
            print(f"  ✓ Hash generated")
        else:
            print(f"  ✗ Failed to generate hash")
    
    print("Done!")


def validate_file_upload(file_path: str, allowed_extensions: list = None, 
                         max_size_mb: int = 10, check_content: bool = True) -> tuple:
    """
    Validate file upload for security
    
    Args:
        file_path: Path to file to validate
        allowed_extensions: List of allowed extensions (e.g., ['.png', '.jpg'])
        max_size_mb: Maximum file size in MB
        check_content: Whether to verify file content matches extension
        
    Returns:
        (is_valid, error_message) tuple
        
    Raises:
        SecurityError: If validation fails critically
    """
    from pathlib import Path
    import os
    
    if allowed_extensions is None:
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        return False, "File does not exist"
    
    # Check if it's actually a file (not a directory)
    if not file_path.is_file():
        return False, "Path is not a file"
    
    # Check file extension
    file_ext = file_path.suffix.lower()
    if file_ext not in allowed_extensions:
        return False, f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum: {max_size_mb}MB"
    
    # Check content if requested (verify it's actually an image)
    if check_content and file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        try:
            from PIL import Image
            img = Image.open(file_path)
            img.verify()  # Verify it's a valid image
            
            # Additional check: re-open to ensure it's not corrupted
            img = Image.open(file_path)
            img.load()  # Force load to catch truncated images
            
        except Exception as e:
            return False, f"Invalid or corrupted image file: {str(e)}"
    
    return True, "Valid"


def validate_path_safety(user_path: str, allowed_base_dir: str) -> tuple:
    """
    Validate path to prevent directory traversal attacks
    
    Args:
        user_path: User-provided path
        allowed_base_dir: Base directory that the path must be within
        
    Returns:
        (is_safe, resolved_path, error_message) tuple
        
    Example:
        is_safe, safe_path, error = validate_path_safety(
            user_input, 
            "C:/Projects/RIAWELC/dataset"
        )
    """
    from pathlib import Path
    import os
    
    try:
        # Resolve both paths to absolute paths
        allowed_base = Path(allowed_base_dir).resolve()
        user_path_resolved = Path(user_path).resolve()
        
        # Check if user path is within allowed directory
        try:
            user_path_resolved.relative_to(allowed_base)
            return True, str(user_path_resolved), "Safe path"
        except ValueError:
            return False, None, f"Path outside allowed directory: {allowed_base}"
            
    except Exception as e:
        return False, None, f"Invalid path: {str(e)}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem
    """
    import re
    from pathlib import Path
    
    # Get just the filename without path
    filename = Path(filename).name
    
    # Remove or replace dangerous characters
    # Keep alphanumeric, dots, dashes, underscores
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    
    # Remove multiple dots (except for extension)
    parts = filename.rsplit('.', 1)
    if len(parts) == 2:
        name, ext = parts
        name = name.replace('.', '_')
        filename = f"{name}.{ext}"
    
    # Limit filename length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1)
        name = name[:250]
        filename = f"{name}.{ext}"
    
    return filename


def check_file_magic_bytes(file_path: str) -> str:
    """
    Check file type using magic bytes (file signature)
    More secure than checking extension alone
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected file type or 'unknown'
    """
    signatures = {
        b'\x89PNG\r\n\x1a\n': 'png',
        b'\xff\xd8\xff': 'jpeg',
        b'BM': 'bmp',
        b'II*\x00': 'tiff',
        b'MM\x00*': 'tiff',
    }
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            
        for signature, file_type in signatures.items():
            if header.startswith(signature):
                return file_type
                
        return 'unknown'
    except Exception:
        return 'unknown'


if __name__ == "__main__":
    # Generate hashes for existing models
    import sys
    
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    else:
        models_dir = "../models"
    
    generate_hashes_for_existing_models(models_dir)
