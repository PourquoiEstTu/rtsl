import torch
from tgcn_model import GCN_muti_att
from configs import Config
import json
import onnxruntime as ort
import os
# from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

PRETRAINED_MODEL = 0
NUM_SAMPLES = 0

def generate_class_integer_mappings(directory: str, mappings_exist: bool, json_path: str, max_classes = None) :
    """
    Generate integer -> class and class -> integer mapping and load them.
    If mappings already exist, they should not be re-generated
    directory: is the output directory if mappings exist; is the input 
        directory when mappings do exist
    """
    if mappings_exist :
        with open(f"{directory}/idx_to_class.json", 'r') as f:
            idx_to_class = json.load(f)
            print("idx_to_class loaded from disk")
        with open(f"{directory}/class_to_idx.json", 'r') as f:
            class_to_idx = json.load(f)
            print("class_to_idx loaded from disk")
    else :
        with open(json_path, "r") as f:
            data = json.load(f)

        # get glosses and filter them down if desired
        all_glosses = sorted([entry["gloss"] for entry in data])
        if max_classes is not None:
            glosses = all_glosses[:max_classes]  # only keep the first max_classes glosses
        else:
            glosses = all_glosses

        class_to_idx = {g: i for i, g in enumerate(glosses)}
        with open(f"{directory}/class_to_idx.json", 'w') as f :
            json.dump(class_to_idx, f, indent=2)
        print("{class: idx} dictionary created")

        idx_to_class = {i: g for i, g in enumerate(glosses)}
        with open(f"{directory}/idx_to_class.json", 'w') as f :
            json.dump(idx_to_class, f, indent=2)
        print("{idx: class} dictionary created")

    return idx_to_class, class_to_idx

# idx_to_class, class_to_idx = generate_class_integer_mappings("/home/pourquoi/repos/rtsl/backend/app/", mappings_exist=True, json_path="/home/pourquoi/repos/rtsl/backend/app/all.json")

# temp implementation
MODEL_LOAD_INFO = {
    "loaded": False,
    "error": None,
    "details": None
}

def _setup_model():
    """Load the ONNX model for inference."""
    
    if ort is None:
        print('ONNX Runtime not available')
        MODEL_LOAD_INFO['error'] = 'ONNX Runtime not installed'
        return
    
    # Get model path - ASL100 (active)
    backend_dir = "../"
    onnx_path = os.path.join(backend_dir, 'models', 'asl100.onnx')
    # ASL2000 model path (commented out)
    # onnx_path = os.path.join(backend_dir, 'models', 'asl2000.onnx')
    
    if not os.path.exists(onnx_path):
        print(f'ONNX model not found at {onnx_path}')
        MODEL_LOAD_INFO['error'] = f'ONNX model not found. Please run convert_to_onnx.py first.'
        return
    
    try:
        providers = ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        pretrained_model = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        
        # Get input shape
        input_shape = pretrained_model.get_inputs()[0].shape
        if len(input_shape) >= 3:
            feature_len = input_shape[2] if input_shape[2] > 0 else 100
            NUM_SAMPLES = feature_len // 2
        
        MODEL_LOAD_INFO['loaded'] = True
        MODEL_LOAD_INFO['details'] = f'Loaded ONNX model from {onnx_path}'
        print(f'ONNX model loaded: {input_shape}, NUM_SAMPLES={NUM_SAMPLES}')
    except Exception as e:
        print(f'Failed loading ONNX model: {e}')
        import traceback
        traceback.print_exc()
        MODEL_LOAD_INFO['loaded'] = False
        MODEL_LOAD_INFO['error'] = str(e)
        pretrained_model = None
    return pretrained_model, NUM_SAMPLES
PRETRAINED_MODEL, NUM_SAMPLES = _setup_model()

# checkpoint_path = "/home/pourquoi/repos/rtsl/backend/app/checkpoints/asl1000/pytorch_model.bin"
# config_path =     "/home/pourquoi/repos/rtsl/backend/app/checkpoints/asl1000/config.ini"
# config = Config(config_path)
#
# # initialize model
# model = GCN_muti_att(
#     input_feature=config.num_samples * 2,  # 50 * 2 = 100
#     hidden_feature=config.hidden_size,      # 256
#     num_class=2000,
#     p_dropout=config.drop_p,               # 0.3
#     num_stage=config.num_stages            # 24
# )
#
# # Load weights
# checkpoint = torch.load(checkpoint_path, map_location='cpu', )
# state_dict = checkpoint.get('state_dict', checkpoint)
# model.load_state_dict(state_dict, strict=False)
# model.eval()
