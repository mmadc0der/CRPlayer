"""
Device Management and Type Safety Utilities

This module provides comprehensive device management and type safety utilities
for the screen page classification pipeline, ensuring proper handling of
CUDA/CPU devices and PyTorch tensor types.
"""

import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any, Tuple
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class DeviceManager:
    """Centralized device management for the classification pipeline."""
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Preferred device ('cuda', 'cpu', or None for auto)
        """
        self.preferred_device = preferred_device
        self.device = self._get_optimal_device()
        self.device_info = self._get_device_info()
        
        logger.info(f"DeviceManager initialized with device: {self.device}")
        logger.info(f"Device info: {self.device_info}")
    
    def _get_optimal_device(self) -> torch.device:
        """Get the optimal device for computation."""
        if self.preferred_device:
            if self.preferred_device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device(self.preferred_device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'current_gpu': torch.cuda.current_device(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'gpu_memory_allocated': torch.cuda.memory_allocated(0),
                'gpu_memory_reserved': torch.cuda.memory_reserved(0),
                'gpu_compute_capability': f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            })
        
        return info
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def move_to_device(self, obj: Any, dtype: Optional[torch.dtype] = None) -> Any:
        """
        Move object to the current device with proper type handling.
        
        Args:
            obj: Object to move (tensor, model, etc.)
            dtype: Optional dtype to convert to
            
        Returns:
            Object moved to device
        """
        if isinstance(obj, torch.Tensor):
            if dtype is not None:
                obj = obj.to(dtype=dtype)
            return obj.to(self.device)
        elif isinstance(obj, nn.Module):
            return obj.to(self.device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(item, dtype) for item in obj)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v, dtype) for k, v in obj.items()}
        else:
            logger.warning(f"Cannot move object of type {type(obj)} to device")
            return obj
    
    def create_tensor(self, data: Any, dtype: torch.dtype = torch.float32, 
                     requires_grad: bool = False) -> torch.Tensor:
        """
        Create a tensor on the current device.
        
        Args:
            data: Data to create tensor from
            dtype: Tensor dtype
            requires_grad: Whether tensor requires gradients
            
        Returns:
            Tensor on current device
        """
        tensor = torch.tensor(data, dtype=dtype, requires_grad=requires_grad)
        return tensor.to(self.device)
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {
            'device': str(self.device)
        }
        
        if torch.cuda.is_available():
            info.update({
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'available_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
            })
        else:
            # For CPU, we can't easily get memory info
            info.update({
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'available_gb': 0.0
            })
        
        return info
    
    def clear_cache(self):
        """Clear CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        model.eval()
        model = model.to(self.device)
        
        # Enable optimizations
        if hasattr(torch, 'jit'):
            try:
                model = torch.jit.optimize_for_inference(model)
                logger.info("Model optimized with TorchScript")
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        return model
    
    def get_optimal_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...], 
                              max_batch_size: int = 64) -> int:
        """
        Find optimal batch size for the given model and device.
        
        Args:
            model: Model to test
            input_shape: Input tensor shape (excluding batch dimension)
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        model.eval()
        model = model.to(self.device)
        
        optimal_batch_size = 1
        
        for batch_size in range(1, max_batch_size + 1):
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                optimal_batch_size = batch_size
                
                # Clear cache
                self.clear_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Optimal batch size found: {optimal_batch_size}")
                    break
                else:
                    raise e
        
        return optimal_batch_size

class TypeSafeModelWrapper:
    """Type-safe wrapper for PyTorch models with device management."""
    
    def __init__(self, model: nn.Module, device_manager: DeviceManager):
        """
        Initialize type-safe model wrapper.
        
        Args:
            model: PyTorch model to wrap
            device_manager: Device manager instance
        """
        self.model = model
        self.device_manager = device_manager
        self.model = self.device_manager.move_to_device(model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with type safety.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Ensure input is on correct device and dtype
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(x)}")
        
        x = self.device_manager.move_to_device(x)
        
        # Ensure model is in eval mode for inference
        self.model.eval()
        
        with torch.no_grad():
            return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with type safety and proper output handling.
        
        Args:
            x: Input tensor
            
        Returns:
            Prediction tensor
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input tensor
            
        Returns:
            Class predictions
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

def get_device_manager(preferred_device: Optional[str] = None) -> DeviceManager:
    """
    Get a global device manager instance.
    
    Args:
        preferred_device: Preferred device ('cuda', 'cpu', or None for auto)
        
    Returns:
        DeviceManager instance
    """
    return DeviceManager(preferred_device)

def ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device, 
                           dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Ensure tensor is on the specified device with proper dtype.
    
    Args:
        tensor: Input tensor
        device: Target device
        dtype: Target dtype
        
    Returns:
        Tensor on target device
    """
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)

def create_dataloader_with_device(dataset, device: torch.device, **kwargs) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with proper device handling.
    
    Args:
        dataset: PyTorch dataset
        device: Target device
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader with device handling
    """
    def collate_fn(batch):
        """Custom collate function to move data to device."""
        data, targets = zip(*batch)
        data = torch.stack(data).to(device)
        targets = torch.tensor(targets).to(device)
        return data, targets
    
    return torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **kwargs)

def save_model_with_device_info(model: nn.Module, path: Union[str, Path], 
                               device_manager: DeviceManager, **kwargs):
    """
    Save model with device information.
    
    Args:
        model: Model to save
        path: Save path
        device_manager: Device manager instance
        **kwargs: Additional save arguments
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'device_info': device_manager.device_info,
        'model_info': {
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        **kwargs
    }
    
    torch.save(save_dict, path)
    logger.info(f"Model saved to {path} with device info")

def load_model_with_device_info(path: Union[str, Path], model: nn.Module, 
                               device_manager: DeviceManager) -> nn.Module:
    """
    Load model with device information.
    
    Args:
        path: Model path
        model: Model to load state dict into
        device_manager: Device manager instance
        
    Returns:
        Loaded model on correct device
    """
    checkpoint = torch.load(path, map_location=device_manager.device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    model = device_manager.move_to_device(model)
    
    # Log device info
    if 'device_info' in checkpoint:
        logger.info(f"Model loaded from {path}")
        logger.info(f"Original device info: {checkpoint['device_info']}")
    
    return model

# Global device manager instance
_global_device_manager = None

def get_global_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager

def set_global_device_manager(device_manager: DeviceManager):
    """Set the global device manager instance."""
    global _global_device_manager
    _global_device_manager = device_manager

# Convenience functions
def get_device() -> torch.device:
    """Get the current device from global device manager."""
    return get_global_device_manager().get_device()

def move_to_device(obj: Any, dtype: Optional[torch.dtype] = None) -> Any:
    """Move object to current device."""
    return get_global_device_manager().move_to_device(obj, dtype)

def clear_cache():
    """Clear CUDA cache."""
    get_global_device_manager().clear_cache()

def get_memory_info() -> Dict[str, float]:
    """Get memory usage information."""
    return get_global_device_manager().get_memory_info()

if __name__ == "__main__":
    # Example usage
    device_manager = DeviceManager()
    print(f"Device: {device_manager.get_device()}")
    print(f"Device info: {device_manager.device_info}")
    print(f"Memory info: {device_manager.get_memory_info()}")
    
    # Test tensor creation
    tensor = device_manager.create_tensor([1, 2, 3, 4])
    print(f"Created tensor: {tensor}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor dtype: {tensor.dtype}")