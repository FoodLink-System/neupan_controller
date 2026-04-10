#!/usr/bin/env python3
"""Export trained DUNE model to ONNX format for C++ inference.

Usage:
  python3 scripts/export_dune_onnx.py config/neupan/checkpoints/<model>.pth config/neupan/dune_diff.onnx

The DUNE model takes obstacle points as input and predicts dual variables (mu)
for collision avoidance. Input: (batch, 2) obstacle points. Output: (batch, n_constraints) mu values.
"""
import sys
import os
import torch
import onnx

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest .pth file in the checkpoint directory."""
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {checkpoint_dir}")
    # Sort by modification time, newest first
    pth_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
    return os.path.join(checkpoint_dir, pth_files[0])


def export_to_onnx(model_path, output_path):
    """Load a trained DUNE PyTorch model and export to ONNX."""
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # The DUNE model architecture: simple MLP that maps (2,) -> (n_constraints,)
    # We need to reconstruct the model architecture from the checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # It might be the model itself
        model = checkpoint
        state_dict = None

    if state_dict is not None:
        # Infer model architecture from state dict
        # DUNE is typically: Linear(2, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, n_out)
        layers = []
        layer_keys = sorted([k for k in state_dict.keys() if 'weight' in k])
        for i, key in enumerate(layer_keys):
            w = state_dict[key]
            in_features = w.shape[1]
            out_features = w.shape[0]
            layers.append(torch.nn.Linear(in_features, out_features))
            if i < len(layer_keys) - 1:  # ReLU between layers, not after last
                layers.append(torch.nn.ReLU())

        model = torch.nn.Sequential(*layers)
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Determine input/output shapes
    # DUNE input: single obstacle point (2,) or batch (N, 2)
    dummy_input = torch.randn(1, 2)

    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"Model input shape: {dummy_input.shape}")
    print(f"Model output shape: {dummy_output.shape}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['obstacle_point'],
        output_names=['mu'],
        dynamic_axes={
            'obstacle_point': {0: 'batch_size'},
            'mu': {0: 'batch_size'},
        },
        opset_version=13,
    )

    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"ONNX model exported to: {output_path}")
    print(f"Input: obstacle_point (batch, 2)")
    print(f"Output: mu (batch, {dummy_output.shape[1]})")

    # Test with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    import numpy as np
    test_input = np.random.randn(5, 2).astype(np.float32)
    result = session.run(None, {'obstacle_point': test_input})
    print(f"ONNX Runtime test: input {test_input.shape} -> output {result[0].shape}")
    print("ONNX export verified successfully!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Auto-find latest checkpoint
        checkpoint_dir = 'config/neupan/checkpoints'
        model_path = find_latest_checkpoint(checkpoint_dir)
        output_path = 'config/neupan/dune_diff.onnx'
    elif len(sys.argv) == 2:
        model_path = sys.argv[1]
        output_path = 'config/neupan/dune_diff.onnx'
    else:
        model_path = sys.argv[1]
        output_path = sys.argv[2]

    print(f"Loading model from: {model_path}")
    export_to_onnx(model_path, output_path)
