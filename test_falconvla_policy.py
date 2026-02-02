#!/usr/bin/env python3
"""Test script to verify FalconVLA policy with Aloha-compatible input/output."""

import numpy as np
from openpi.policies import falconvla_policy

def test_falconvla_transforms():
    """Test that FalconVLA transforms work with Aloha-compatible data."""
    
    # Create a sample input that matches Aloha format
    sample_data = falconvla_policy.make_falconvla_example()
    
    print("Original input data:")
    print(f"  State shape: {sample_data['state'].shape}")
    print(f"  Images: {list(sample_data['images'].keys())}")
    print(f"  Prompt: {sample_data['prompt']}")
    
    # Test input transform
    input_transform = falconvla_policy.FalconVLAInputs()
    transformed_input = input_transform(sample_data)
    
    print("\nTransformed input:")
    print(f"  Image keys: {list(transformed_input['image'].keys())}")
    print(f"  Image mask keys: {list(transformed_input['image_mask'].keys())}")
    print(f"  State shape: {transformed_input['state'].shape}")
    print(f"  Prompt: {transformed_input.get('prompt', 'N/A')}")
    
    # Verify standard image keys exist
    assert "base_0_rgb" in transformed_input["image"]
    assert "left_wrist_0_rgb" in transformed_input["image"]
    assert "right_wrist_0_rgb" in transformed_input["image"]
    print("\n✓ All expected image keys present")
    
    # Test output transform with dummy actions
    dummy_actions = np.random.randn(25, 14)  # action_horizon=25, action_dim=14
    output_data = {"actions": dummy_actions}
    
    output_transform = falconvla_policy.FalconVLAOutputs()
    transformed_output = output_transform(output_data)
    
    print(f"\nOutput actions shape: {transformed_output['actions'].shape}")
    assert transformed_output['actions'].shape == (25, 14)
    print("✓ Output transform works correctly")
    
    print("\n✅ All tests passed! FalconVLA policy is Aloha-compatible.")

if __name__ == "__main__":
    test_falconvla_transforms()
