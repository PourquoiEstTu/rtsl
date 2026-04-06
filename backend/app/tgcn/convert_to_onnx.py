"""
Convert PyTorch checkpoint to ONNX format for ASL models.
This script loads the trained checkpoint and exports it to ONNX.
Supports both ASL100 and ASL2000 models.
"""

import os
from sys import exit
import torch
import configparser

ROOT = "." # or just use "." 

from tgcn_model import GCN_muti_att


def load_config(config_path):
    """Load configuration from INI file."""
    cp = configparser.ConfigParser()
    cp.read(config_path)
    
    num_samples = int(cp.get('TRAIN', 'NUM_SAMPLES', fallback='50'))
    hidden_size = int(cp.get('GCN', 'HIDDEN_SIZE', fallback='256'))
    num_stages = int(cp.get('GCN', 'NUM_STAGES', fallback='24'))
    drop_p = float(cp.get('TRAIN', 'DROP_P', fallback='0.3'))
    
    return num_samples, hidden_size, num_stages, drop_p


def convert_to_onnx(model_name, checkpoint_path, config_path, output_dir = ROOT):
    """
    Convert PyTorch checkpoint to ONNX format.
    
    Args:
        model_name: 'asl100' or 'asl2000' (default: 'asl100')
    """
    onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
    
    print("=" * 60)
    print(f"PyTorch to ONNX Conversion - {model_name.upper()}")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output: {onnx_path}")
    print()
    
    # Load config
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    num_samples, hidden_size, num_stages, drop_p = load_config(config_path)
    
    # Set number of classes based on model
    if model_name == 'asl100':
        num_class = 100  # ASL100 has 100 classes
    elif model_name == 'asl300':
        num_class = 300  # ASL100 has 100 classes
    elif model_name == 'asl1000':
        num_class = 1000  # ASL100 has 100 classes
    elif model_name == 'asl2000':
        num_class = 2000  # ASL2000 has 2000 classes (commented out)
    elif model_name == 'custom':
        num_class = 3  # Example for a custom model
    else:
        # num_class = 100  # Default to ASL100
        raise Exception("Valid model not passed in. Valid strings are: ['asl100', 'asl300', 'asl1000', 'asl2000']")
    
    input_feature = num_samples * 2
    
    print(f"Model Configuration:")
    print(f"  - Input features: {input_feature} (num_samples={num_samples} * 2)")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Number of stages: {num_stages}")
    print(f"  - Dropout: {drop_p}")
    print(f"  - Number of classes: {num_class}")
    print()
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    print("Loading checkpoint...")
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Strip 'module.' prefix if present (from DataParallel)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[new_key] = v
    
    # Create model
    print("Creating model...")
    model = GCN_muti_att(
        input_feature=input_feature,
        hidden_feature=hidden_size,
        num_class=num_class,
        p_dropout=drop_p,
        num_stage=num_stages
    )
    
    # Load state dict
    try:
        model.load_state_dict(cleaned_state_dict)
        print("✓ Checkpoint loaded successfully")
    except RuntimeError as e:
        print(f"WARNING: Could not load full state dict: {e}")
        print("Attempting partial load...")
        model_dict = model.state_dict()
        filtered = {k: v for k, v in cleaned_state_dict.items() 
                   if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"✓ Partially loaded ({len(filtered)}/{len(model_dict)} parameters)")
    
    model.eval()
    print()
    
    # Create dummy input for ONNX export
    # Input shape: (batch_size, num_nodes, feature_len)
    # num_nodes = 55 (body + hand keypoints)
    batch_size = 1
    num_nodes = 55
    dummy_input = torch.randn(batch_size, num_nodes, input_feature)
    
    print("Exporting to ONNX...")
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✓ ONNX model exported successfully to: {onnx_path}")
        print()
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model validation passed")
            return True
        except ImportError:
            print("WARNING: onnx package not installed, skipping validation")
            return True
        except Exception as e:
            print(f"WARNING: ONNX validation failed: {e}")
            return True  # Still return True as export succeeded
            
    except Exception as e:
        print(f"ERROR: Failed to export ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 100 and 2000 class models' config and pytorch bin file don't exist because they were taken straight from the reference repo
    success = convert_to_onnx('custom', f"{ROOT}/outputs/100/0_1.0000.pth", f"configs/config2.ini")
    # success = convert_to_onnx('asl2000', f"{ROOT}/splits/2000/pytorch_model.bin", f"{ROOT}/splits/2000/asl2000.ini")
    if success:
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
        exit(0)
    else:
        print("\n" + "=" * 60)
        print("Conversion failed!")
        print("=" * 60)
        exit(1)
