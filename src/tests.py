import torch
import pytest
from src.model import MNISTClassifier

def test_parameter_count():
    """Test that the model has less than 20k parameters"""
    model = MNISTClassifier()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 20000, f"Model has {param_count} parameters, which exceeds the 20k limit"

def test_batch_normalization_usage():
    """Test that the model uses batch normalization"""
    model = MNISTClassifier()
    has_batch_norm = False
    
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            has_batch_norm = True
            break
    
    assert has_batch_norm, "Model does not use batch normalization"

def test_dropout_usage():
    """Test that the model uses dropout"""
    model = MNISTClassifier()
    has_dropout = False
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            has_dropout = True
            break
    
    assert has_dropout, "Model does not use dropout"

def test_gap_no_fc():
    """Test that the model uses Global Average Pooling and no fully connected layers"""
    model = MNISTClassifier()
    has_gap = False
    has_fc = False
    
    for module in model.modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            has_gap = True
        if isinstance(module, torch.nn.Linear):
            has_fc = True
    
    assert has_gap, "Model does not use Global Average Pooling"
    assert not has_fc, "Model should not use fully connected layers"

def test_model_output():
    """Test that the model produces correct output shape"""
    model = MNISTClassifier()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
